# AICT-Pothole-Detection

## 📜 코드 실행 순서

1. **합성 데이터셋 생성 (YOLO)**
  - mask.py: 도로 이미지에서 아스팔트 영역 마스크를 추출한다.
  - synthesize_pothole.py: 추출된 마스크 위에 포트홀 이미지를 합성하고, YOLO 형식의 바운딩 박스(Bbox) 라벨을 생성한다.

1. **학습 데이터셋 준비 (Detectron2)**
  - generate_sam_annotations.py: 원본 데이터셋의 Bbox 주석을 SAM을 활용해 Polygon 주석으로 변환한다.
  - make_annotations.py: SAM으로 생성된 주석 파일(annotations_sam.json)을 기반으로 증강된 데이터셋(annotations_augmented.json)을 위한 주석 파일을 만든다.
  - augment_dataset.py: 원본 이미지의 아스팔트 색상을 변경하여 데이터셋을 증강하고, 이에 맞는 주석 파일을 생성한다.

2. **모델 학습**
   - train_mask_rcnn.py: 준비된 원본 데이터셋과 증강된 데이터셋을 모두 사용해 Mask R-CNN 모델을 훈련한다.

3. **모델 평가 및 예측**
   - evaluate_model.py: 훈련이 완료된 모델의 성능을 COCO metrics(mAP)를 이용해 정량적으로 평가하고 시각화한다.
   - predict_test.py: 학습된 모델을 사용하여 새로운 이미지에 대한 포트홀을 예측하고 결과를 시각화한다.

## 📝 코드별 상세 설명
### mask.py
- 역할: 도로 이미지에서 아스팔트 영역 마스크를 추출한다.
- 작동 방식:
    1. BGR과 HSV 색상 공간을 기반으로 회색조, 밝기, 채도 등의 조건을 활용해 아스팔트 영역을 정밀하게 분리하고, 이를 이진 마스크 파일로 저장한다.
 
### synthesize_pothole.py
- 역할: mask.py에서 추출한 도로 마스크 위에 포트홀 이미지를 합성하고, 그 위치를 YOLO 형식의 라벨로 자동 생성한다.
- 작동 방식:
    1. 마스크 내의 유효한 위치를 찾아 포트홀 이미지의 크기, 밝기, 회전 등을 무작위로 조절한 후, 페더링(feathering) 기법을 사용해 자연스럽게 합성한다.
    2. 합성된 포트홀의 바운딩 박스를 계산하여 라벨 파일로 저장합니다.

### generate_sam_annotations.py
- 역할: Bounding Box(Bbox) 형태의 원본 주석을 Segment Anything Model(SAM)을 사용하여 더욱 정교한 Polygon(다각형) 주석으로 변환하는 역할을 한다.
- 작동 방식:
    1. 원본 COCO 주석 파일(annotations.json)을 읽는다.
    2. SAM 모델을 로드하고, 각 이미지에 대해 Bbox 주석을 SAM의 입력으로 사용한다.
    3. SAM이 생성한 마스크(Mask)를 OpenCV 함수를 이용해 폴리곤 형태로 변환한다.
    4. 변환된 폴리곤 주석과 함께 area 및 bbox 값을 재계산하여 새로운 주석 파일(annotations_sam.json)을 만든다.
- 체크 사항:
    - 이 코드는 Mask R-CNN 모델의 성능을 향상시키기 위해 주석의 품질을 높이는 전처리 단계이다.
    - SAM 모델 체크포인트 경로(sam_checkpoint)와 데이터셋 경로(base_dir)를 올바르게 설정해야 한다.

### make_annotations.py
- 역할: generate_sam_annotations.py에서 생성된 annotations_sam.json 파일을 복사하고, 이미지 파일 경로만 images_augmented 폴더를 가리키도록 수정한다.
- 작동 방식:
    1. 원본 annotations_sam.json 파일을 읽는다.
    2. images 리스트를 순회하며 각 이미지 정보의 file_name 속성을 새로운 증강 이미지의 경로(images_augmented/파일이름_aug.jpg)로 변경한다.
    3. 경로가 변경된 새 JSON 파일을 annotations_augmented.json으로 저장한다.
- 체크 사항:
    - augment_dataset.py의 주석 생성 로직이 복잡해질 수 있어, 이 스크립트는 주석 파일의 경로만 간단하게 변경하는 역할을 분리한 것이다. 이는 코드의 가독성을 높이고 디버깅을 용이하게 한다.

### augment_dataset.py
- 역할: 학습 데이터셋의 양을 늘리기 위해 색상 변경을 통한 데이터 증강을 수행한다. 특히 아스팔트 도로의 색상을 인위적으로 변경하여 다양한 환경에 대한 모델의 강건성(robustness)을 높이는 목적이다.
- 작동 방식:
    1. change_asphalt_color 함수에서 BGR 및 HSV 색상 공간의 특정 조건(회색조, 밝기, 채도 등)을 기반으로 아스팔트 영역을 식별하는 마스크를 생성한다.
    2. 이때, generate_sam_annotations.py에서 생성된 포트홀의 폴리곤 주석을 제외하여 포트홀 영역이 변색되지 않도록 한다.
    3. 식별된 아스팔트 영역의 색상을 무작위로(red, white, yellow, ochre) 변경한다.
    4. 새로 생성된 이미지와 주석 정보를 포함한 annotations_augmented.json 파일을 생성한다.
- 체크 사항:
    - 이 코드는 모델의 일반화 성능을 높이기 위한 증강 기술이다.
    - 이미지 경로(base_dir)를 올바르게 설정해야 한다.

### train_mask_rcnn.py
- 역할: 데이터셋을 등록하고 Detectron2의 Mask R-CNN 모델을 학습시킨다.
- 작동 방식:
    1. register_datasets 함수를 통해 원본 SAM 주석 데이터셋(pothole_train_sam)과 증강 데이터셋(pothole_train_augmented)을 모두 학습 데이터셋으로 등록한다.
    2. Detectron2 설정(cfg)을 로드하고, 모델 가중치, 학습 반복 횟수(MAX_ITER), 배치 크기, 클래스 수 등 하이퍼파라미터를 설정한. 특히 total_images 변수를 사용하여 에포크에 맞춰 MAX_ITER를 계산한다.
    3. Wandb(Weights & Biases)를 사용하여 학습 과정의 손실 값, 성능 지표 등을 실시간으로 로깅한다.
    4. DefaultTrainer를 상속받은 WandbTrainer를 사용하여 학습을 시작한다.
- 체크 사항:
    - 이 스크립트가 파이프라인의 핵심 학습 부분이다.
    - cfg.DATASETS.TRAIN에 pothole_train_sam과 pothole_train_augmented를 모두 포함시켜 두 데이터셋을 함께 학습시키는 것이 중요하다.
    - OUTPUT_DIR과 wandb.init의 name을 프로젝트에 맞게 변경해야 한다.

### evaluate_model.py
- 역할: 학습된 모델의 성능을 정량적으로 평가하고 그 결과를 시각화한다.
- 작동 방식:
    1. register_coco_instances를 통해 검증 데이터셋(pothole_val)을 등록한다.
    2. 학습된 모델의 가중치 파일(model_final.pth)을 로드하여 DefaultPredictor를 초기화한다.
    3. COCOEvaluator를 사용하여 COCO metrics(mAP)를 계산한다.
    4. matplotlib과 seaborn을 사용하여 결과를 히트맵으로 시각화하고, Wandb에 숫자 결과와 함께 차트 이미지를 로깅다.
- 체크 사항:
    - model_weights_path와 output_dir을 올바르게 설정하는 것이 중요하다.
 
  ### predict_test.py
- 역할: 학습된 모델을 사용하여 테스트 이미지에 대한 포트홀 예측 결과를 시각화한다.
- 작동 방식:
    1. 학습된 모델의 가중치를 로드하고 DefaultPredictor를 초기화한다.
    2. 지정된 테스트 이미지 폴더의 모든 이미지를 순회하며, 각 이미지에 대해 예측을 수행한다.
    3. Visualizer를 사용하여 예측된 마스크와 Bbox를 원본 이미지 위에 오버레이하여 시각적인 결과물을 생성한다.
    4. 결과 이미지를 지정된 출력 폴더에 저장한다.
- 체크 사항:
    - cfg.MODEL.WEIGHTS, test_image_dir, output_dir 경로를 프로젝트에 맞게 변경해야 한다.
 
## 전체 파이프라인 요약
1. mask.py 실행 (도로 영역 마스크 생성)
2. synthesize_pothole.py 실행 (도로 마스크 위에 포트홀 합성)
3. generate_sam_annotations.py 실행 (Bbox -> Polygon 주석 변환)
4. make_annotations.py 실행 (증강 데이터셋용 주석 파일 경로 생성)
5. augment_dataset.py 실행 (이미지 색상 증강 및 최종 주석 파일 생성)
6. train_mask_rcnn.py 실행 (원본 + 증강 데이터셋으로 모델 학습)
7. evaluate_model.py 실행 (학습된 모델 성능 평가)
8. predict_test.py 실행 (새 이미지에 대한 예측 결과 시각화)
