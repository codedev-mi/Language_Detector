import h5py
import json
import os

MODEL_PATH = os.path.join(
    "dataset",
    "sign_language_mobilenet.h5"
)

with h5py.File(MODEL_PATH, "r+") as model_file:

    raw_config = model_file.attrs["model_config"]

    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")

    config = json.loads(raw_config)

    fixed_layers = 0

    def fix_config(obj):

        global fixed_layers

        if isinstance(obj, dict):

            # Fix InputLayer configuration
            if (
                obj.get("class_name") == "InputLayer"
                and isinstance(obj.get("config"), dict)
            ):

                layer_config = obj["config"]

                if "batch_shape" in layer_config:

                    layer_config["batch_input_shape"] = (
                        layer_config.pop("batch_shape")
                    )

                    fixed_layers += 1

            # Continue recursively
            for value in obj.values():
                fix_config(value)

        elif isinstance(obj, list):

            for item in obj:
                fix_config(item)

    fix_config(config)

    if fixed_layers == 0:

        print("No InputLayer batch_shape configurations found.")

    else:

        updated_config = json.dumps(config)

        # Recreate attribute using variable-length UTF-8
        del model_file.attrs["model_config"]

        model_file.attrs.create(
            "model_config",
            updated_config,
            dtype=h5py.string_dtype(encoding="utf-8")
        )

        print(
            f"Successfully fixed {fixed_layers} InputLayer configuration(s)."
        )

print("Model compatibility patch completed.")
