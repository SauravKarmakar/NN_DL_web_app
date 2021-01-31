from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import os.path
import pandas as pd


app = Flask(__name__)
# Orginial models shipped.
regressor_model = load_model("models/part1_regressor_model.h5")
classifier_model = load_model("models/part2_classifier_model.h5")
file_available = False
count = 1


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/import_data", methods=["POST"])
def import_data():
    """
    For importing the "Signals" data, from file
    """
    value = [x for x in request.form.values()]
    # print(value)
    global file_available
    if str(value[0]).lower() == "signals.csv":
        if os.path.isfile("signals.csv"):  # check the file existence
            file_available = True
        else:
            file_available = False
    else:
        file_available = False
        return render_template(
            "index.html",
            data_imported='Unable to find the mentioned file.!!, Please use "Signals.csv" to make it work :)',
        )
    return render_template("index.html", data_imported="Data Imported.")


@app.route("/import_target", methods=["POST"])
def import_target():
    """
    Importing the target column from the data set
    """
    target_available = False
    value = [x for x in request.form.values()]
    print(str(value[0]).lower())
    global file_available
    if file_available != True:
        return render_template(
            "index.html",
            target_imported="Dats is not imported yet !!!",
        )

    if str(value[0]).lower() == "signal_strength":
        df = pd.read_csv("Signals.csv")
        columns_lst = df.columns.values.tolist()
        is_present = ["Signal_Strength" in columns_lst]
        if is_present:
            target_available = True
        else:
            target_available = False
    else:
        return render_template(
            "index.html",
            target_imported='Unable to find the mentioned target column.!!, Please use "Signal_Strength" which is there in the dataset. :)',
        )
    return render_template(
        "index.html", data_imported="Data Imported.", target_imported="Target Found."
    )


@app.route("/nn_reg_train", methods=["POST"])
def nn_reg_train():
    """
    Train the NN Regressor Model
    """
    id = request.args.get("id", default=0, type=int)
    global regressor_model
    if id == 1:
        # regressor_model = tf.keras.Sequential(
        #     [
        #         layers.Dense(4, activation="relu"),
        #         layers.Dense(4, activation="relu"),
        #         layers.Dense(4, activation="relu"),
        #         layers.Dense(1),
        #     ]
        # )

        # regressor_model.compile(
        #     loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam()
        # )

        nnr_trained = regressor_model.to_json()
        return render_template(
            "index.html",
            nnr_trained="NN Regressor Model Trained. Model: {}".format(nnr_trained),
        )
    elif id == 2:
        global count
        regressor_model.save("pickels/part1_regressor_model" + str(count) + ".h5")
        count += 1
        return render_template(
            "index.html",
            nnr_saved="Model Saved. - 'part1_regressor_model{}.h5'".format(count),
        )


@app.route("/nn_class_train", methods=["POST"])
def nn_class_train():
    """
    Train the NN Classifier Model
    """
    id = request.args.get("id", default=0, type=int)
    global classifier_model
    if id == 1:
        nnc_trained = classifier_model.to_json()
        return render_template(
            "index.html",
            nnc_trained="NN Classifier Model Trained. Model: {}".format(nnc_trained),
        )
    elif id == 2:
        global count
        classifier_model.save("pickels/part2_classifier_model" + str(count) + ".h5")
        count += 1
        return render_template(
            "index.html",
            nnc_saved="Model Saved. - 'part2_classifier_model{}.h5'".format(count),
        )


if __name__ == "__main__":
    app.run(debug=True)
