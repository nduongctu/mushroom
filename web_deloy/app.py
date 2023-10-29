import flask
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

with open('model/rf_mushroom_pre.joblib', 'rb') as f:
    model = load(f)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        cap_shape = flask.request.form['cap-shape']
        cap_surface = flask.request.form['cap-surface']
        cap_color = flask.request.form['cap-color']
        bruises = flask.request.form['bruises']
        odor = flask.request.form['odor']
        gill_attachment = flask.request.form['gill-attachment']
        gill_spacing = flask.request.form['gill-spacing']
        gill_size = flask.request.form['gill-size']
        gill_color = flask.request.form['gill-color']
        stalk_shape = flask.request.form['stalk-shape']
        stalk_surface_above_ring = flask.request.form['stalk-surface-above-ring']
        stalk_surface_below_ring = flask.request.form['stalk-surface-below-ring']
        stalk_color_above_ring = flask.request.form['stalk-color-above-ring']
        stalk_color_below_ring = flask.request.form['stalk-color-below-ring']
        veil_color = flask.request.form['veil-color']
        ring_number = flask.request.form['ring-number']
        ring_type = flask.request.form['ring-type']
        spore_print_color = flask.request.form['spore-print-color']
        population = flask.request.form['population']
        habitat = flask.request.form['habitat']

        input_variables = pd.DataFrame([[cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment,
                                         gill_spacing, gill_size, gill_color, stalk_shape,
                                         stalk_surface_above_ring, stalk_surface_below_ring,
                                         stalk_color_above_ring, stalk_color_below_ring,
                                         veil_color, ring_number, ring_type, spore_print_color,
                                         population, habitat]],
                                       columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                                                'stalk-shape', 'stalk-surface-above-ring',
                                                'stalk-surface-below-ring', 'stalk-color-above-ring',
                                                'stalk-color-below-ring', 'veil-color', 'ring-number',
                                                'ring-type', 'spore-print-color', 'population', 'habitat'],
                                       dtype='int64', index=['input'])

        label_encoder = LabelEncoder()
        for column in input_variables.columns:
            input_variables[column] = label_encoder.fit_transform(input_variables[column])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html',
                                     original_input={'Cap_shape': cap_shape, 'Cap_surface': cap_surface,
                                                     'Cap_color': cap_color, 'Bruises': bruises, 'Odor': odor,
                                                     'Gill_attachment': gill_attachment,
                                                     'Gill_spacing': gill_spacing, 'Gill_size': gill_size,
                                                     'Gill_color': gill_color, 'Stalk_shape': stalk_shape,
                                                     'Stalk_surface_above_ring': stalk_surface_above_ring,
                                                     'Stalk_surface_below_ring': stalk_surface_below_ring,
                                                     'Stalk_color_above_ring': stalk_color_above_ring,
                                                     'Stalk_color_below_ring': stalk_color_below_ring,
                                                     'Veil_color': veil_color, 'Ring_number': ring_number,
                                                     'Ring_type': ring_type, 'Spore_print_color': spore_print_color,
                                                     'population': population, 'Habitat': habitat},
                                     result=predictions)


if __name__ == '__main__':
    app.run()