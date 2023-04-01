from .generate_schemas import gen
from dotenv import load_dotenv


load_dotenv()
gen()
from .import_kinetic_models import main
main()
