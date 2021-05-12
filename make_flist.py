# 원본   이미지 flist 경로: ./flist/room_flist.flist
# 지워진 이미지 flist 경로: ./flist/input_flist.flist
# 마스크 이미지 flist 경로: ./flist/mask_flist.flist
import os
import glob

image_path_list = ['./datasets/train/', './datasets/val/', './datasets/test/', './datasets/train_mask/', './datasets/val_mask/', './datasets/test_mask/']                               # 이미지 파일 경로 리스트
flist_path_list = ['./datasets/train.flist', './datasets/val.flist', './datasets/test.flist', './datasets/train_mask.flist', './datasets/val_mask.flist', './datasets/test_mask.flist'] # flist 파일 경로 리스트

class FlistGenerator():
    def __init__(self, image_path_list, flist_path_list):
        self.image_path_list = image_path_list
        self.flist_path_list = flist_path_list

        for i in range(6):
            with open(flist_path_list[i], 'w') as f:                                                                # flist 파일 쓰기모드로 열기
                file_name_list = [os.path.basename(k) for k in glob.glob(image_path_list[i]+'*')]                   # 이미지 파일 경로 리스트로 만들기
                file_len = len(file_name_list)                                                                      # 파일 개수

                for j in range(file_len):
                    data = os.path.join(image_path_list[i], file_name_list[j]) + '\n'                               # 이미지 파일 경로\n
                    f.write(data)                                                                                   # flist 파일에 이미지 파일 경로\n 쓰기

flist_generator = FlistGenerator(image_path_list, flist_path_list)
flist_generator._generate_flist()