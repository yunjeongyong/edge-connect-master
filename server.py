from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from main import main
from PIL import Image
from scipy.misc import imresize, imsave
import time

app = Flask(__name__)
api = Api(app)

# /api/images 경로에 대한 연결을 처리하는 클래스
class ImageUpload(Resource):

    # post 메소드
    def post(self):
        try:
            # 프론트엔드로부터 전달받은 파라미터 파싱
            parser = reqparse.RequestParser()
            parser.add_argument('image', type=FileStorage, location='files')
            args = parser.parse_args()

            # 사용자가 수정하려는 원본 이미지
            image = args['image']

            # 이미지의 이름을 새로 설정하기 위해 확장자 추출
            ext = image.filename[image.filename.rfind('.'):]
            # 이미지 이름 형식 설정 ('im' + timestamp)
            filename = 'im' + str(int(time.time() * 1000)).zfill(15) + ext

            # 이미지가 저장될 경로 설정
            image_path = './static/{0}'.format(filename)
            # 해당 경로에 이미지 저장
            image.save(image_path)

            # 성공 여부 및 이미지 경로 리턴
            return {
                'success': True,
                'image': image_path
            }
        except Exception as e:
            # print(str(e))
            # 실행 도중 에러가 났을 경우 에러 메시지 리턴
            return {
                'success': False,
                'error': str(e)
            }

# /api/maskeds 경로에 대한 연결을 처리하는 클래스
class MaskedImageUpload(Resource):

    # post 메소드
    def post(self):
        try:
            # parse parameters
            parser = reqparse.RequestParser()
            parser.add_argument('masked', type=FileStorage, location='files')
            parser.add_argument('mask', type=FileStorage, location='files')
            parser.add_argument('filename', type=str)
            args = parser.parse_args()

            # 마스크 된 이미지 'masked'
            image = args['masked']
            # 마스크 이미지 'mask'
            mask = args['mask']
            # 원본 이미지
            name = args['filename']

            # 파일 확장자 추출
            ext = name[name.rfind('.'):]
            # 확장자를 제외한 이름 추출
            standard_name = name[:name.rfind('.')]
            # 마스크 된 이미지 파일명을 '원본파일명_masked'로 설정
            filename = standard_name + '_masked' + ext
            # 마스크 이미지 파일명을 '원본파일명_mask'로 설정
            mask_name = standard_name + '_mask' + ext

            # save input images
            image_path = './static/{0}'.format(filename)
            image.save(image_path)

            # save mask image
            mask_path = './static/{0}'.format(mask_name)
            mask.save(mask_path)

            result_path = './static/results/{0}'.format(filename)

            # resize images
            # 리사이징을 위해 Image.open으로 저장된 파일을 RGB로 연 뒤, imresize 함수를 통해 리사이징
            resizes = [256, 256]
            img_masked = Image.open(image_path).convert('RGB')
            img_masked = imresize(img_masked, resizes)
            imsave(image_path, img_masked)

            # 마스크에 대해서도 리사이징 작업
            img_mask = Image.open(mask_path).convert('RGB')
            img_mask = imresize(img_mask, resizes)
            imsave(mask_path, img_mask)

            # 딥러닝 코드 호출
            main(mode=2, model=3, input=image_path, mask=mask_path)

            # 딥러닝 연산 후 성공 여부, 원본 이미지 경로, 마스크 이미지 경로, 결과 이미지 경로 리턴
            return {
                'success': True,
                'image': image_path,
                'result': result_path,
                'mask': mask_path
            }
        except Exception as e:
            # 에러가 났을 경우 에러 메시지 리턴
            return {
                'success': False,
                'error': str(e)
            }

# Flask_RESTful 라이브러리를 통해 각 클래스들에 경로 설정
api.add_resource(ImageUpload, '/api/images')
api.add_resource(MaskedImageUpload, '/api/maskeds')

if __name__ == '__main__':
    # Flask를 통해 서버 실행
    app.run(debug=True)