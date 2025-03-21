import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory # 문제 1 불러와야할 모델의 경로를 잘못잡음

# 피클로 저장된 모델 로드
with open('/home/potato/DX-02/workspace/HW5-preceptron_ANN/ann_penumonia.keras', 'rb') as f:
    model = pickle.load(f)

# 이미지가 저장된 폴더 경로
image_dir = '/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/train/PNEUMONIA'

# 폴더 내 모든 이미지 파일을 순회
for img_name in os.listdir(image_dir):
    # 파일 확장자가 .jpeg인 경우에만 처리
    if img_name.endswith('.jpeg'):
        img_path = os.path.join(image_dir, img_name)  # 이미지 파일의 전체 경로
        img = image.load_img(img_path, target_size=(64, 64))  # 이미지 크기 조정
        img_array = image.img_to_array(img)  # 이미지를 넘파이 배열로 변환
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
        img_array = img_array / 255.0  # 정규화

        # 예측
        predictions = model.predict(img_array)
        print(f"Predictions for {img_name}: {predictions}")

        # 예측 결과가 클래스 확률이라면, 가장 높은 확률을 가진 클래스를 선택
        predicted_class = np.argmax(predictions, axis=1)  # 예측된 클래스
        print(f"Predicted class for {img_name}: {predicted_class}")

        # 이미지 출력
        plt.imshow(img)
        plt.axis('off')  # 축 없애기
        plt.show()
# ==================================감자 방식 실패


# # ==================================교수님방식 이미지 랜덤 로드
# img_height = 64
# img_width = 64
# batch_size = 32

# def prepare(ds, suffle = False, augment = False):
#     ds = ds.map(
#         lambda x, y : (tf.image.resize(x, [img_height, img_width]), y)
#         )
#     return ds

# model = tf.keras.models.load_model('ann_penumonia.keras')
# model.summary()

# test_folder = '/home/potato/DX-02/workspace/HW5-preceptron_ANN/chest_xray/test/'
# test_n = test_folder + 'NORMAL/'
# test_p = test_folder + 'PENUMIA/'

# test_ds = image_dataset_from_directory(test_folder,
#                                        validation_split = 0.2,
#                                        subset = "validation",
#                                        seed=123,
#                                        image_size=(img_height, img_width ),
#                                        batch_size = batch_size)

# test_ds = prepare(test_ds)
# test_img, test_label = next(iter(test_ds))
# test_img = np.array(test_img)
# test_label = np.array(test_label)
# predict = model(test_img)
# plt.figure()
# for idx in range(0, len(test_img)-1):
#     plt.subplot(5,5,idx+1)
#     print(test_label[idx])
#     print(predict[idx])
#     plt.title('actual = ', test_label[idx], 'predict = ', predict[idx])
#     plt.imshow(test_img[idx])
# plt.show()
# # ==================================교수님방식 이미지 랜덤 로드



# # ===========================================그래프 출력 항목

# # Accuracy
# plt.plot(ann_model.history['accuracy'])
# plt.plot(ann_model.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training set', 'Validationset'], loc='upper left')
# plt.savefig('train_accuracy.png')
# plt.show(block=False)
# plt.clf()

# # Loss
# plt.plot(ann_model.history['val_loss'])
# plt.plot(nn_model.history['loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training set', 'Test set'],loc='upper left')
# plt.savefig('train_loss.png')
# plt.show(block=False)
# plt.clf()