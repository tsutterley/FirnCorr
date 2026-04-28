"""Create a table of available SMB and firn models"""

import pathlib
from FirnCorr.io import load_database

# documentation directory
directory = pathlib.Path(__file__).parent
# load the database of SMB and firn models
models = load_database()

# create model table
for region in ["ais", "gris"]:
    # filter models by region
    models_table = directory.joinpath("_assets", f"{region}-models.csv")
    fid = models_table.open(mode="w", encoding="utf8")
    # write to csv
    fid.write("Model,Directory\n")
    for model, parameters in models.items():
        if region not in parameters:
            continue
        # extract the model directory
        if isinstance(parameters[region]["model_file"], str):
            model_directory = pathlib.Path(
                parameters[region]["model_file"]
            ).parent
        elif isinstance(parameters[region]["model_file"], list):
            model_directory = pathlib.Path(
                parameters[region]["model_file"][0]
            ).parent
        # extract the reference
        reference = parameters.get("reference", None)
        # write the model and directory to the csv file
        if reference is not None:
            fid.write(
                f"`{model} <{reference}>`_,``<model_path>/{model_directory}``\n"
            )
        else:
            fid.write(f"{model},``<model_path>/{model_directory}``\n")
    # close the file
    fid.close()
