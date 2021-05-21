from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from main import main
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
            standard_name = name[:name.rfind('.')]
            filename = standard_name + '_masked' + ext
            mask_name = standard_name + '_mask' + ext

            image_path = './static/{0}'.format(filename)
            image.save(image_path)

            mask_path = './static/{0}'.format(mask_name)
            mask.save(mask_path)

            result_path = './static/{0}_result{1}'.format(standard_name, ext)

            main(mode=2, model=3, checkpoints='./checkpoints/places2', input=image_path, mask=mask_path, output=result_path)

            return {
                'success': True,
                'image': image_path,
                'result': result_path,
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

