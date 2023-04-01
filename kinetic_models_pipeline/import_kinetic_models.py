from collections import Counter
import os
import re
from dateutil import parser
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Tuple, Union

import habanero
import requests
from pydantic import ValidationError
import rmgpy
from dotenv import load_dotenv
from rmgpy import kinetics as rmgkinetics, constants
from rmgpy.molecule import Molecule
from rmgpy.reaction import Reaction as RmgReaction
from rmgpy.species import Species as RmgSpecies
from rmgpy.data.kinetics.library import KineticsLibrary
from rmgpy.data.thermo import ThermoLibrary
from rmgpy.thermo import NASA, ThermoData, Wilhoit, NASAPolynomial

from kinetic_models_pipeline.models import Arrhenius, ArrheniusEP, KineticModel, Kinetics, Reaction, ReactionSpecies, Source, Author, Thermo, Transport, Species, Isomer, Structure, NamedSpecies

class ModelDir(NamedTuple):
    name: str
    thermo_path: Path
    kinetics_path: Path
    source_path: Path


class EnvironmentVariableMissing(Exception):
    pass


def get_model_paths(data_path: Path, ignore_list: List[str] = []) -> Iterable[ModelDir]:
    for path in data_path.rglob("*"):
        name = path.name
        thermo_path = path / "RMG-Py-thermo-library" / "ThermoLibrary.py"
        kinetics_path = path / "RMG-Py-kinetics-library" / "reactions.py"
        source_path = path / "source.txt"
        paths = [thermo_path, kinetics_path, source_path]

        if path.is_dir() and name not in ignore_list and all(p.exists() for p in paths):
            yield ModelDir(
                name=name,
                thermo_path=thermo_path,
                kinetics_path=kinetics_path,
                source_path=source_path
            )

def create_test_kinetic_model() -> KineticModel:
    structure = Structure(adjlist="", smiles="", multiplicity=0)
    isomer = Isomer(formula="", inchi="", structures=[structure])
    species = Species(isomers=[isomer])
    named_species = NamedSpecies(name="", species=species)
    author = Author(firstname="kian", lastname="mehrabani")
    source = Source(doi="", publication_year=0, title="", journal_name="", journal_volume="", page_numbers="", authors=[author])
    transport = Transport(species=species, geometry=0, well_depth=0, collision_diameter=0, dipole_moment=0, polarizability=0, rotational_relaxation=0, source=source)
    kinetic_model = KineticModel(name="", named_species=[named_species], transport=[transport], source=source)

    return kinetic_model


class MissingAuthorData(Exception):
    pass


class InvalidAuthorData(Exception):
    pass

class ThermoLibraryLoadError(Exception):
    pass

class KineticsLibraryLoadError(Exception):
    pass

class CreateSpeciesError(Exception):
    pass

class CreateSourceError(Exception):
    pass


class DOIError(Exception):
    pass


def create_authors(author_entries: Iterable[dict]) -> Iterable[Author]:
    if author_entries is None:
        raise MissingAuthorData()
    for entry in author_entries:
        if entry["given"] is None or entry["family"] is None:
            raise InvalidAuthorData(entry)
        yield Author(firstname=entry["given"], lastname=entry["family"])


def get_doi(source_path: Path):
    """
    Get the DOI from the source.txt file
    """

    with open(source_path, "r") as f:
        source = f.read()

    regex = re.compile(r"10.\d{4,9}/\S+")
    matched_list = regex.findall(source)
    matched_list = [d.rstrip(".") for d in matched_list]
    # There are sometimes other trailing characters caught up, like ) or ]
    # We should probably clean up the source.txt files
    # But let's try cleaning them here.

    def clean(doi):
        for opening, closing in ["()", "[]"]:
            if doi.endswith(closing):
                if doi.count(closing) - doi.count(opening) == 1:
                    # 1 more closing than opening
                    # remove the last closing
                    doi = doi[:-1]
        return doi

    matched_list = [clean(d) for d in matched_list]
    matched_set = set(matched_list)

    if len(matched_set) == 0:
        raise DOIError(f"DOI not found for path: {source_path}")
    elif len(matched_set) > 1:
        raise DOIError(f"Found multiple DOIS: {matched_set}")
    else:
        return matched_list[0]


def create_source(path: Path) -> Source:
    try:
        crossref = habanero.Crossref(mailto="kianmehrabani@gmail.com")
        doi = get_doi(path)
        reference = crossref.works(ids=doi).get("message", "") if doi else {}
        created_info = reference.get("created", {})
        date = parser.parse(created_info.get("date-time", "")) if created_info else None
        year = date.year if date else ""
        title_body = reference.get("title", "")
        source_title = title_body[0] if isinstance(title_body, list) else title_body
        name_body = reference.get("short-container-title", "")
        journal_name = name_body[0] if isinstance(name_body, list) else name_body
        volume_number = reference.get("volume", "")
        page_numbers = reference.get("page", "")
        author_data = reference.get("author")
        authors = create_authors(author_data)

        return Source(
            doi=doi,
            publication_year=year,
            title=source_title,
            journal_name=journal_name,
            journal_volume=volume_number,
            page_numbers=page_numbers,
            authors=list(authors),
        )
    except (InvalidAuthorData, DOIError, ValidationError) as e:
        raise CreateSourceError(e)


def create_structure(molecule: Molecule) -> Structure:
    return Structure(
        smiles=molecule.to_smiles(),
        adjlist=molecule.to_adjacency_list(),
        multiplicity=molecule.multiplicity,
    )


def create_species(molecule: Molecule) -> Species:
    formula = molecule.get_formula()
    inchi = molecule.to_inchi()

    structure = create_structure(molecule)
    isomer = Isomer(formula=formula, inchi=inchi, structures=[structure])

    return Species(isomers=[isomer])


def create_named_species(name: str, molecule: Molecule) -> NamedSpecies:
    return NamedSpecies(name=name, species=create_species(molecule))


def create_nested_species(rmg_species: RmgSpecies) -> NamedSpecies:
    formula = rmg_species.molecule[0].get_formula()
    try:
        inchi = rmg_species.get_augmented_inchi()
    except (IndexError, AttributeError):
        raise CreateSpeciesError()

    structures = [create_structure(m) for m in rmg_species.molecule]
    isomer = Isomer(formula=formula, inchi=inchi, structures=structures)

    species = Species(isomers=[isomer])

    return NamedSpecies(name=rmg_species.label, species=species)


def create_thermo(path: Path, source: Source) -> Iterable[Tuple[Thermo, NamedSpecies]]:
    local_context = {
        "ThermoData": ThermoData,
        "Wilhoit": Wilhoit,
        "NASAPolynomial": NASAPolynomial,
        "NASA": NASA,
    }
    library = ThermoLibrary(label=str(path))
    library.SKIP_DUPLICATES = True
    try:
        library.load(path, local_context=local_context)
    except (rmgpy.exceptions.DatabaseError, ValueError) as e:
        raise ThermoLibraryLoadError(e)
    for species_name, entry in library.entries.items():
        species = create_named_species(species_name, entry.item)
        thermo_data = entry.data
        poly1, poly2 = thermo_data.polynomials
        thermo = Thermo(
            species=species.species,
            polynomial1=poly1.coeffs.tolist(),
            polynomial2=poly2.coeffs.tolist(),
            min_temp1=poly1.Tmin.value_si,
            max_temp1=poly1.Tmax.value_si,
            min_temp2=poly2.Tmin.value_si,
            max_temp2=poly2.Tmax.value_si,
            source=source,
        )

        yield thermo, species


def create_reaction(rmg_reaction: RmgReaction) -> Tuple[Reaction, List[NamedSpecies]]:
    all_species: List[RmgSpecies] = [*rmg_reaction.reactants, *rmg_reaction.products]
    species_counts = Counter(all_species)
    rs = []
    named_species = []
    for r in rmg_reaction.reactants:
        ns = create_nested_species(r)
        named_species.append(ns)
        coeff = species_counts[r]
        reaction_species = ReactionSpecies(species=ns.species, coefficient=-coeff)
        rs.append(reaction_species)

    for p in rmg_reaction.products:
        ns = create_nested_species(p)
        named_species.append(ns)
        coeff = species_counts[p]
        reaction_species = ReactionSpecies(species=ns.species, coefficient=coeff)
        rs.append(reaction_species)

    reaction = Reaction(reaction_species=rs, reversible=rmg_reaction.reversible)

    return reaction, named_species


def create_kinetics_data(rmg_kinetics_data) -> Union[Arrhenius, ArrheniusEP]:
    return Arrhenius(a=0, a_si=0, a_units="", n=0, e=0, e_si=0, e_units="", s="")

def create_kinetics(path: Path, source: Source) -> Iterable[Tuple[Kinetics, List[NamedSpecies]]]:
    local_context = {
        "KineticsData": rmgkinetics.KineticsData,
        "Arrhenius": rmgkinetics.Arrhenius,
        "ArrheniusEP": rmgkinetics.ArrheniusEP,
        "MultiArrhenius": rmgkinetics.MultiArrhenius,
        "MultiPDepArrhenius": rmgkinetics.MultiPDepArrhenius,
        "PDepArrhenius": rmgkinetics.PDepArrhenius,
        "Chebyshev": rmgkinetics.Chebyshev,
        "ThirdBody": rmgkinetics.ThirdBody,
        "Lindemann": rmgkinetics.Lindemann,
        "Troe": rmgkinetics.Troe,
        "R": constants.R,
    }
    library = KineticsLibrary(label=str(path))
    library.SKIP_DUPLICATES = True
    try:
        library.load(path, local_context=local_context)
    except rmgpy.exceptions.DatabaseError as e:
        raise KineticsLibraryLoadError(e)
    for entry in library.entries.values():
        rmg_kinetics_data = entry.data
        rmg_reaction = entry.item
        reaction, species = create_reaction(rmg_reaction)
        kinetics_data = create_kinetics_data(rmg_kinetics_data)

        min_temp = getattr(rmg_kinetics_data.Tmin, "value_si", None)
        max_temp = getattr(rmg_kinetics_data.Tmax, "value_si", None)
        min_pressure = getattr(rmg_kinetics_data.Pmin, "value_si", None)
        max_pressure = getattr(rmg_kinetics_data.Pmax, "value_si", None)

        kinetics = Kinetics(
            reaction=reaction,
            data=kinetics_data,
            for_reverse=False,
            min_temp=min_temp,
            max_temp=max_temp,
            min_pressure=min_pressure,
            max_pressure=max_pressure,
            source=source,
        )

        yield kinetics, species


def group_submodels(thermo_species: Iterable[Tuple[Thermo, NamedSpecies]], kinetics_species: Iterable[Tuple[Kinetics, List[NamedSpecies]]]) -> Tuple[List[Thermo], List[Kinetics], List[NamedSpecies]]:
    thermo = []
    kinetics = []
    species = []

    for t, s in thermo_species:
        thermo.append(t)
        species.append(s)

    for k, s in kinetics_species:
        kinetics.append(k)
        species.extend(s)

    return thermo, kinetics, species


def create_kinetic_model(model_dir: ModelDir) -> Optional[KineticModel]:

    try:
        source = create_source(model_dir.source_path)
        thermo_species = create_thermo(model_dir.thermo_path, source)
        kinetics_species = create_kinetics(model_dir.kinetics_path, source)
        thermo, kinetics, species = group_submodels(thermo_species, kinetics_species)

        return KineticModel(
            name=model_dir.name,
            named_species=species,
            thermo=thermo,
            kinetics=kinetics,
            source=source,
        )
    except (CreateSourceError, ThermoLibraryLoadError, KineticsLibraryLoadError, CreateSpeciesError):
        return None


def import_rmg_models(endpoint: str, data_path: Path = Path("/rmg-models")) -> None:
    model_dirs = get_model_paths(data_path)
    for model_dir in model_dirs:
        km = create_kinetic_model(model_dir)
        if km is not None:
            requests.post(endpoint, data=km.json(exclude_none=True, exclude_unset=True))

def main():
    endpoint = os.getenv("POST_ENDPOINT")
    rmg_models_path = os.getenv("RMG_MODELS_PATH")
    if endpoint is None:
        raise EnvironmentVariableMissing("POST_ENDPOINT not set")

    if rmg_models_path is not None:
        import_rmg_models(endpoint, data_path=Path(rmg_models_path))
    else:
        import_rmg_models(endpoint)
