# generated by datamodel-codegen:
#   filename:  openapi.json
#   timestamp: 2023-04-01T05:45:36+00:00

from __future__ import annotations

from typing import List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class Arrhenius(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    a: float = Field(..., title='A')
    a_si: float = Field(..., title='A Si')
    a_delta: Optional[float] = Field(None, title='A Delta')
    a_units: str = Field(..., title='A Units')
    n: float = Field(..., title='N')
    e: float = Field(..., title='E')
    e_si: float = Field(..., title='E Si')
    e_delta: Optional[float] = Field(None, title='E Delta')
    e_units: str = Field(..., title='E Units')
    s: str = Field(..., title='S')


class ArrheniusEP(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    a: float = Field(..., title='A')
    a_si: float = Field(..., title='A Si')
    a_units: float = Field(..., title='A Units')
    n: float = Field(..., title='N')
    e0: float = Field(..., title='E0')
    e0_si: float = Field(..., title='E0 Si')
    e0_units: str = Field(..., title='E0 Units')


class Author(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    firstname: str = Field(..., title='Firstname')
    lastname: str = Field(..., title='Lastname')


class Source(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    doi: str = Field(..., title='Doi')
    publication_year: int = Field(..., title='Publication Year')
    title: str = Field(..., title='Title')
    journal_name: str = Field(..., title='Journal Name')
    journal_volume: str = Field(..., title='Journal Volume')
    page_numbers: str = Field(..., title='Page Numbers')
    authors: List[Author] = Field(..., min_items=1, title='Authors')
    prime_id: Optional[str] = Field(None, title='Prime Id')


class Structure(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    adjlist: str = Field(..., title='Adjlist')
    smiles: str = Field(..., title='Smiles')
    multiplicity: int = Field(..., title='Multiplicity')


class ValidationError(BaseModel):
    loc: List[Union[str, int]] = Field(..., title='Location')
    msg: str = Field(..., title='Message')
    type: str = Field(..., title='Error Type')


class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]] = Field(None, title='Detail')


class Isomer(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    formula: str = Field(..., title='Formula')
    inchi: str = Field(..., title='Inchi')
    structures: List[Structure] = Field(..., min_items=1, title='Structures')


class Species(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    isomers: List[Isomer] = Field(..., min_items=1, title='Isomers')
    cas_number: Optional[str] = Field(None, title='Cas Number')
    prime_id: Optional[str] = Field(None, title='Prime Id')


class Thermo(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    species: Species
    min_temp1: float = Field(..., title='Min Temp1')
    max_temp1: float = Field(..., title='Max Temp1')
    min_temp2: float = Field(..., title='Min Temp2')
    max_temp2: float = Field(..., title='Max Temp2')
    source: Source
    preferred_key: Optional[str] = Field(None, title='Preferred Key')
    reference_temp: Optional[float] = Field(None, title='Reference Temp')
    reference_pressure: Optional[float] = Field(None, title='Reference Pressure')
    enthalpy_formation: Optional[float] = Field(None, title='Enthalpy Formation')
    prime_id: Optional[str] = Field(None, title='Prime Id')
    polynomial1: List[float] = Field(..., max_items=7, min_items=7, title='Polynomial1')
    polynomial2: List[float] = Field(..., max_items=7, min_items=7, title='Polynomial2')


class Transport(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    species: Species
    geometry: float = Field(..., title='Geometry')
    well_depth: float = Field(..., title='Well Depth')
    collision_diameter: float = Field(..., title='Collision Diameter')
    dipole_moment: float = Field(..., title='Dipole Moment')
    polarizability: float = Field(..., title='Polarizability')
    rotational_relaxation: float = Field(..., title='Rotational Relaxation')
    source: Source
    prime_id: Optional[str] = Field(None, title='Prime Id')


class NamedSpecies(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    name: str = Field(..., title='Name')
    species: Species


class ReactionSpecies(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    coefficient: int = Field(..., title='Coefficient')
    species: Species


class Reaction(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    reaction_species: List[ReactionSpecies] = Field(
        ..., min_items=1, title='Reaction Species'
    )
    reversible: bool = Field(..., title='Reversible')
    prime_id: Optional[str] = Field(None, title='Prime Id')


class Kinetics(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    reaction: Reaction
    data: Union[Arrhenius, ArrheniusEP] = Field(..., title='Data')
    source: Source
    for_reverse: bool = Field(..., title='For Reverse')
    uncertainty: Optional[float] = Field(None, title='Uncertainty')
    min_temp: Optional[float] = Field(None, title='Min Temp')
    max_temp: Optional[float] = Field(None, title='Max Temp')
    min_pressure: Optional[float] = Field(None, title='Min Pressure')
    max_pressure: Optional[float] = Field(None, title='Max Pressure')
    prime_id: Optional[str] = Field(None, title='Prime Id')


class KineticModel(BaseModel):
    id: Optional[UUID] = Field(None, title='Id')
    name: str = Field(..., title='Name')
    named_species: List[NamedSpecies] = Field(..., min_items=1, title='Named Species')
    kinetics: Optional[List[Kinetics]] = Field([], title='Kinetics')
    thermo: Optional[List[Thermo]] = Field([], title='Thermo')
    transport: Optional[List[Transport]] = Field([], title='Transport')
    source: Source
    prime_id: Optional[str] = Field(None, title='Prime Id')
