from flask import Flask, render_template, request
from samgeo.text_sam import LangSAM
import os

app = Flask(__name__)

# Define directory paths
app_dir = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get form data
        image_path = str(request.form['image_path'])
        text_prompt = str(request.form['text_prompt'])
        box_threshold = float(request.form['box_threshold'])
        text_threshold = float(request.form['text_threshold'])

        # Construct the full path to the image file
        image_full_path = os.path.join(app_dir, image_path)         

        # Define output filename for raster
        raster_output_filename = request.form.get('raster_filename', 'output_raster').strip() or "output"


        # Initialize LangSAM model
        sam = LangSAM()

    
        # Predict using LangSAM model
        sam.predict(image_full_path, text_prompt, box_threshold, text_threshold)

        # Show annotations
        sam.show_anns(
            cmap='Greys_r',
            add_boxes=False,
            alpha=1,
            title='Automatic Segmentation',
            blend=False,
            output=os.path.join(app_dir, raster_output_filename + '.tif'),  # Using the defined output filename
        )

        # Define output filename for vector
        vector_output_filename = request.form.get('vector_filename', 'output_vector').strip() or "vector"

        # Convert raster to vector
        sam.raster_to_vector(os.path.join(app_dir, raster_output_filename + ".tif"), os.path.join(app_dir, vector_output_filename + ".shp"))

        # Return success message
        return render_template('success.html')
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
