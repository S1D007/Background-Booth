from flask import Flask, request, render_template, jsonify
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
import PIL.Image
import io
import os
import tempfile
import base64
import requests

app = Flask(__name__)

seg_net = TracerUniversalB7(device='cpu', batch_size=1)
fba = FBAMatting(device='cpu', input_tensor_size=4 * 1024, batch_size=1)
trimap = TrimapGenerator()
preprocessing = PreprocessingStub()
postprocessing = MattingMethod(matting_module=fba, trimap_generator=trimap, device='cpu')
interface = Interface(pre_pipe=preprocessing, post_pipe=postprocessing, seg_pipe=seg_net)

def save_image_temp(image, filename):
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, filename)
    image.save(filepath, format='PNG')
    return filepath

@app.route('/remove_background', methods=['POST'])
def remove_background_endpoint():
    try:
        image_file = request.files['image']
        new_bg_file = request.files['bg']
        overlay_image = request.files['overlay']
        image_bytes = image_file.read()
        new_bg_bytes = new_bg_file.read()
        overlay_image_bytes = overlay_image.read()
        result_image = interface([PIL.Image.open(io.BytesIO(image_bytes))])[0]

        new_bg_image = PIL.Image.open(io.BytesIO(new_bg_bytes))

        new_bg_image = new_bg_image.resize(result_image.size, PIL.Image.FIXED)

        final_result_image = PIL.Image.new('RGBA', result_image.size, (255, 255, 255, 0))

        final_result_image.paste(new_bg_image, (0, 0), new_bg_image)

        final_result_image.paste(result_image, (0, 0), result_image)

        filename = f'{next(tempfile._get_candidate_names())}.png'
        result_path = save_image_temp(final_result_image, filename)

        with open(result_path, 'rb') as data:
            file_data = data.read()
            file_base64 = base64.b64encode(file_data).decode('utf-8')
            overlay_base64 = base64.b64encode(overlay_image_bytes).decode('utf-8')
            payload = {
                'base': file_base64,
                'overlay': overlay_base64,
                'uploadFor': "test_123"
            }
            response = requests.post("https://s39nhbwtx9.execute-api.ap-south-1.amazonaws.com/template", json=payload)
            print(response)
            url = response.json().get('image')
            return jsonify({'result_url': url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view_result/<filename>')
def view_result(filename):
    return render_template('view_result.html', result_url=f'/static/results/{filename}')

@app.route('/')
def index():
    return render_template('./index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
