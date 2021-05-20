from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
api = Api(app)


# @api.route('/api/images')
class ImageUpload(Resource):

    def get(self):
        return {}

    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('image', type=FileStorage, location='files')
            args = parser.parse_args()

            image = args['image']

            ext = image.filename[image.filename.rfind('.'):]
            filename = 'im' + str(int(time.time() * 1000)).zfill(15) + ext

            image_path = './static/{0}'.format(filename)
            image.save(image_path)

            return {
                'success': True,
                'image': image_path
            }
        except Exception as e:
            # print(str(e))
            return {
                'success': False,
                'error': str(e)
            }

class MaskedImageUpload(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('masked', type=FileStorage, location='files')
            parser.add_argument('mask', type=FileStorage, location='files')
            parser.add_argument('filename', type=str)
            args = parser.parse_args()

            print(args)

            image = args['masked']
            mask = args['mask']
            name = args['filename']

            ext = name[name.rfind('.'):]
            filename = name[:name.rfind('.')] + '_masked' + ext
            maskname = name[:name.rfind('.')] + '_mask' + ext

            image_path = './static/{0}'.format(filename)
            image.save(image_path)

            mask_path = './static/{0}'.format(maskname)
            mask.save(mask_path)

            return {
                'success': True,
                'image': image_path,
                'mask': mask_path
            }
        except Exception as e:
            # print(str(e))
            return {
                'success': False,
                'error': str(e)
            }

api.add_resource(ImageUpload, '/api/images')
api.add_resource(MaskedImageUpload, '/api/maskeds')

if __name__ == '__main__':
    app.run(debug=True)

