
<h1 align="center"> 
  <br> 
  코딩 교구 제작 최종 보고서
</h1>

<h3 align="center">
</h3>  
<p align="right">
  <br>
  프로젝트 기반 빅데이터 서비스 개발자 양성 과정 4기  
  <br>박준식⠂이찬녕⠂이형석⠂임유하</br>
</p>   

## 1. 프로젝트 목적
&nbsp; 코딩 교구는 프로그래밍과 컴퓨터 과학을 학습하고 이해하기 위한 도구, 자원 또는 교육 자료를 말한다. 코딩 교구를 이용해 학생들은 복잡한 문제를 분석하고 해결하며 문제 해결 능력 강화, 자기 주도 학습을 이끌 수 있다. 또한, 현대 사회에서 중요한 역할을 하는 디지털 기술을 배우며 디지털 시대에 필요한 미디어, 소프트웨어, 웹 애플리케이션 등을 이해할 수 있어 지속적인 기술 발전에 대응할 수 있는 능력을 키울 수 있다.
결론적으로 코딩 교구는 디지털 시대로의 변화에 대비하는데 중요한 역할을 하기 때문에 어린 학생들의 미래를 준비하고 다양한 학습 경험 제공을 위해 필요가 있다.

&nbsp; 이에 따라 숫자, 색상, 모양 등 다양한 개념을 이해하는 인지 발달의 시기를 거치는 5세 유아를 대상으로 하는 코딩 교구를 제작하고자 한다.
코딩 교구를 통해 유아들은 놀이로 새로운 개념과 기술을 탐구하면서 호기심을 충족시킬 수 있으며, 움직이면서 학습하는 경향이 있는 유아들이 물체를 움직이거나 상호 작용하면서 체험 중심의 학습을 경험할 수 있다. 또한, 짧은 시간의 집중력을 지닌다는 특징을 가진 유아들에게 놀이 중심의 코딩 교구는 짧은 시간 동안에도 유아들의 주의를 집중시키고 학습을 즐겁게 할 수 있다.

&nbsp; 따라서, 5세 유아의 특징을 고려하여 시각적이고 상호 작용이 있는 놀이를 중심으로 한 코딩 교구를 제공하는 것은 유아의 발달을 존중하고 성장을 적절하게 지원하는 방법이 될 것이다.

## 2. 요구사항 및 목표
<b>[요구 사항]</b>
블록기반 프로그래밍, 이미지 인식 기술

<b>블록 기반 프로그래밍 : 시각적인 블록 요소를 조합하여 코드를 생성</b>
<br>

- 시각적 프로그래밍 블록 : 프로그램의 논리와 동작을 나타내는 시각적 블록. 각각 특정 기능이나 명령을 나타냄<br>
- 논리의 시각적 표현 : 논리적인 개념을 시각적으로 표현. 이를 이해하고 조립<br>
- 실시간 피드백 제공<br>
- 비교적 낮은 진입 장벽 : 텍스트 기반의 프로그래밍보다 진입 장벽이 낮기 때문에 코드를 작성하고 실행하기 비교적 쉬움<br>
- 교육용 목적 : 프로그래밍의 기본 개념을 배우고 이해하는데 활용

<b>이미지 인식 기술 : 디지털 이미지에서 패턴, 특징, 물체 등을 자동으로 감지하고 인식하는 기술</b>
<br>

- 자동화 및 높은 처리 속도 : 자동화되어 있으며, 컴퓨터 비전 알고리즘을 사용하여 빠르게 이미지 처리하고 인식 가능<br>
- 강력한 물체 검출 및 분류 : 객체의 크기, 모양, 색상 등을 기반으로 정확한 물체 식별 가능<br>
- 실시간 처리 : 실시간으로 이미지 처리하고 응답 가능<br>

<b>[최종 목표]</b>
<br>

<b>이미지 인식을 활용한 코딩 문제 풀이 모바일 어플리케이션 제작</b><br>

i) 게임 요소와 시각적 효과를 통해 학습을 더욱 흥미롭게 만들어 학습자가 게임을 플레이 하면서 코딩 개념을 익힐 수 있다.<br>
- 게임 기반 학습 : 코딩을 게임 형식으로 학습하는 방법 포함<br>
- 카메라 연동(카메라 API 및 라이브러리) : 카메라 설정, 화면 크기, 노출 조절, 화질 설정, 캡처 기능 제공

ii) 사용자에게 효과적이고 흥미로운 코딩 학습 경험을 제공<br>
- 자동 피드백 및 개인화된 학습
- 최신 기술 및 트렌드 반영


-> 저연령대를 대상으로 하고, 이미지 인식 모델을 이용해 각 블록 마다 명령어와 같은 역할을 하는 코딩 교구를 기획 및 제작. 효율적인 학습과 차별화된 학습 경험을 쌓는 것을 목표로 함


## 3. 개발 과정
<b>[기획 단계]
<br>
<br>
블록 요소를 조합한 코드를 사진으로 촬영하고 앱에서 실행 버튼을 누르면, 화면 내에서 상호작용이 일어난다.</b>
<br>

- 시각적 프로그래밍 블록의 조합으로 논리적인 개념을 시각적으로 표현할 수 있도록 한다.<br>
- 카메라 촬영 및 터치 스크린 조작으로 상호작용이 일어나게 한다.<br>
- 블록 코딩 교구로 직접 코드를 조합할 수 있도록 한다.<br>
- 모바일 앱 어플리케이션으로 코딩 교구를 사용할 수 있도록 한다.<br>
- 간단한 순서와 패턴을 인식할 수 있도록 한다.<br>

| ![timeline](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/1dfbd6a0-9397-439f-aa4c-2c4603c58761) |
|:--:|
| <b> [Figure1] Project Timeline </b> |


<b>[구현 단계]</b>


| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/2e6ea901-0848-4107-a2f8-e19afe0a84cc) |
|:--:|
| <b> [Figure2] Project Workflow </b> |
| 모든 구성원은 전 과정에 종합적으로 기여했으나, 주도적으로 맡은 역할을 각자의 프레임에 넣어 나타냈다 |

<b>1. 화살표 인식 모델 및 S3 탐구</b>
- 사용자는 웹캠을 통해 화살표를 촬영하고, 시스템은 YOLO를 사용하여 방향을 분류하고, 촬영된 데이터는 Dynamo DB와 S3에 저장되어 추후 분석 및 기록을 제공

i) 데이터 생성:
<br>
- 화살표 이미지에서 컨투어를 추출하고, 무작위 회전 및 이동을 적용하여 40,000개의 학습 데이터를 생성
- 생성된 데이터를 학습하여 딥러닝 모델을 구축했으나 성능이 좋지 않아서 YOLO를 사용

ii) 모델 훈련:
<br>
- YOLO를 사용하여 화살표 방향을 분류하는 모델 훈련

iii) 웹캠을 통한 방향 분류:
<br>
- OpenCV를 사용하여 웹캠으로 사용자가 그린 화살표 이미지를 촬영, 박스 모양을 표시하여 사용자가 정확한 위치에 화살표를 배치하도록 유도
- 박스 안의 이미지만을 추출하여 YOLO 모델을 사용하여 화살표의 방향을 분류

iv) 데이터베이스 구축:

- AWS Dynamo DB를 활용하여 사용자가 촬영한 화살표 이미지와 관련된 정보를 저장하는 데이터베이스를 생성
- 이미지 데이터는 AWS S3에 저장하여 Dynamo DB에는 해당 이미지의 메타데이터를 저장

v) 보유 기간 설정:

- 사용자가 촬영한 이미지를 일정 기간 동안 보유하기 위해 Dynamo DB에 생성 일자, 시간, 모델이 분류한 화살표 방향 등을 저장

<b>2. 모델링</b>

i) 블럭 개수 구하기
| ![blocks](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/859c26b0-278f-41b9-87f3-d9dc73714d06) |
|:--:|
| <b> [Figure3] 블럭 개수 구하기 </b> |
| 가장 큰 컨투어의 비율을 통해 블럭의 개수를 구할 수 있다. |
  
ii) 화살표, 숫자 컨투어 구하기
| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/1fbf72fe-56a8-4c5b-9e4e-fedff5fcbc13) |
|:--:|
| <b> [Figure4] 컨투어 위치 구하기 </b> |
| 블럭의 개수를 파악하고 내부 비율에 따라 화살표와 숫자 컨투어가 있는 위치를 대략적으로 추정 가능함. 해당 지역 근방에 중심이 위치한 컨투어 중 가장 작은 컨투어를 지정하고, 행동의 경우 블럭의 색이 다른 것을 이용해 특정 위치 픽셀의 RGB 값을 추출 |

iii) 학습 이미지 생성
<br>
| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/35e57d7c-cf15-4135-83af-6287fe87e62a) |
|:--:|
| <b> [Figure5] 학습 이미지 생성 </b> |
| 사진을 수천 장 촬영하여 학습 데이터를 구하는 것은 비효율적이어서 직접 생성함. 크기 조절과 무작위 회전, 회색조, 이진화 등의 과정을 거쳐 학습 데이터 생성. 데이터의 수는 화살표는 방향별로 1만 장, 숫자는 숫자별로 2만 장의 데이터를 생성함 |

iv) Train History
<br>
| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/af9124a4-2f8b-4699-8e96-25c3875659a6) |
|:--:|
| <b> [Figure6] Train History </b> |
| 학습 데이터의 다양성이 높지는 않기 때문에 train history의 accuracy는 1에 가깝게 나타남.
-> 정해진 상황에서는 좋은 예측 성능을 보여줄 수 있다. |
<br>

v) 행동 예측
<br>
| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/b5ddf1cc-b03e-43b6-a3fc-a3391a2a1c7f) |
|:--:|
| <b> [Figure7] HSV값 통한 행동 예측 </b> |
| 블럭의 행동을 맞추는 과정. RGB는 3개의 값을 이용해서 추정해야하기 때문에 복잡하고 불편하다. HSV로 바꾸면 H 값 하나로 색상을 알 수 있어, H 값의 조건에 따라 행동을 추정 |

<br>
<br>
<b>3. YOLO 모델 튜닝</b>
<br>
<br>
- roboflow 사이트로 데이터셋 전처리, Hugging Face 사이트로 인식 모델 테스트
<br>
<br>

i) 데이터 레포지토리 생성하기
| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/fd4e4f3c-ac75-4541-af87-be8233fa01a2) |
|:--:|
| <b> [Figure8] 레포지토리 생성 </b> |
| Create New Project 클릭 후, Project Type - Object Detection 선택 |
<br>
<br>

ii) 이미지 데이터 라벨링
| ![labeling](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/d23dfb57-f502-466b-8b48-21a86d2d1c8a) |
|:--:|
| <b> [Figure9] 이미지 라벨링 </b> |
| 훈련 모델에 사용할 이미지 데이터를 가져온 후, 라벨링할 부분에 드래그한 후 클래스 추가<br>모든 사진의 라벨링 작업 후 Add # image to Dataset 클릭한 후에, Train, Valid, Test 비율 설정 (Train 70%, Valid 20%, Test 10% 비율로 설정)|
<br>
<br>
iii) 데이터셋 내보내기

| ![dataset](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/8f03b3f0-3397-45e1-9a6d-19c1a9db82d9) |
|:--:|
| <b> [Figure10] 데이터셋 내보내기 </b> |
| Custom Train and Upload - Get Snippet 으로 zip 파일로 데이터셋 내보내기 |
<br>
<br>
iv) Google Colab 실행 후 모델 학습 실행하기

- Google Colab 실행 후, 런타임 유형을 GPU로 변경<br>
- YOLOv8 설치 및 import<br>
- roboflow에서 라벨링한 데이터셋 가져온 후 모델 학습하기 (에포크 개수 25, 이미지 픽셀 800)<br>
- 학습된 모델은 content/runs/detect/train/weights/best.pt에 저장<br>
- best.pt : 학습된 모델 중 가장 정확도가 높은 값을 나타내는 모델<br>
<br>
<br>
v) 학습 모델을 이용한 화살표, 숫자 인식 테스트 생성 - Hugging Face

| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/a21bb8db-eead-4dba-8682-7b5e11ed76c5) |
|:--:|
| <b> [Figure11] 테스트 - Hugging Face </b> |
| 학습된 yolo 모델을 gradio 모듈을 이용해서 테스트 생성<br>학습된 yolo 모델 가져오기 및 이미지 input 박스 생성 코드 작성후 app.py에 저장<br>필요한 모듈을 requirements.txt에, yolo 모델 라벨값을 values.py에 작성 후 저장<br>Hugging Face에 Space 생성 후, Files에 app.py, best.pt, requirements.txt, values.py 업로드 후 app 클릭 -> 테스트 박스 생성 |

<br>
<br>
vi) 생성된 박스를 이용해 모델 학습

| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/689c1677-9967-4fd2-b1e6-0e4f6529f8b3) |
|:--:|
| <b> [Figure12] YOLO 컨투어 이미지 추출 </b> |
| YOLO를 이용해서 컨투어를 그린 후, 컨투어 속의 이미지 예측을 위해 이미지를 추출 |
<br>
<br>
vii) YOLO 컨투어 이미지 학습

| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/200580a5-06ee-446f-9881-6a1496b0bc44) |
|:--:|
| <b> [Figure13] YOLO 컨투어 이미지 학습 </b> |
| YOLO의 컨투어는 자체 모델에 비해 여백이 많이 잡히므로 무작위 확대와 무작위 이동 과정을 추가하여 학습 데이터를 생성함 |
<br>
<br>
<b>4. App 개발</b>
<br>
<br>
&nbsp; Python 내에서 Mobile Application을 구현하는 것을 목표로 했지만, 기능적 제약이나 환경 설정 문제로 Flutter를 활용해 UI 제작, API 및 DB 연동을 구현했다.
<br>
<br>
초기 기능 구현

- 카메라<br>
- 이동 버튼

추가 기능
- UI 변경<br>
- Database 연동<br>
- AWS 서버 구축 및 API 연동 <br>
<br>
<br>
<b>5. Database 구축</b>
<br>
<br>
i) Cloud Firestore

| ![firebase](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/80083967-6d56-48e2-99e1-398c5a498974) |
|:--:|
| <b> [Figure14] Cloud Firestore </b> |
| 빠른 쿼리와 실시간 동기화 기능을 제공하는 NoSQL Database. Realtime Database에 비해 다양한 기능이 있다고 하지만, 제작한 모바일 앱을 Virtual Machine이 아닌 개인 기기에서 실행했을 때 데이터 적재가 느린 경우가 있어 실험 단계에서만 사용했다. |
<br>
<br>
ii) Realtime Database

| ![realtime_database](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/9b369a7e-e80b-460d-ad70-7a9465160cae) |
|:--:|
| <b> [Figure15] Realtime Database </b> |
| Firebase 초기 Database이지만, 모바일 앱 내에서 찍은 사진이 실시간으로 서버를 거쳐 나와야 하는 상황에서는 실험 결과 속도가 더 빨랐기 때문에 최종적으로 Realtime Database를 선택했다. |
<br>
<br>
<b>6.  API 배포</b>

| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/22e6da6d-82c7-4623-b2f0-f18117472439) |
|:--:|
| <b> [Figure16] App, DB, Server 도식화 </b> |
| AWS EC2 내에 모델 실행 파일을 넣고 FastAPI를 이용해서 API를 배포<br>App에서 사진을 촬영하면 이미지가 Firebase의 Realtime Database에 저장되고 다시 API를 호출하여 요청을 보내면 API는 Database에서 이미지를 가져와서 예측 결과를 App으로 반환 |

<br>
<br>

## 4. 최종 결과물
<br>
<br>
i) YOLO

| ![YOLO](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/f3d77404-16f5-445d-989c-6318d29bf4e8) |
|:--:|
| <b> [Figure17] YOLO 결과 </b> |
| YOLO로 생성한 이미지 인식 모델은 화살표와 숫자를 각각 인식하고, 인식률은 0.7을 나타내었다. |
<br>
<br>
ii) YOLO Tuning 모델 / Osmo 블럭 인식 결과

| ![image](https://github.com/bigdata4th-first-line/block_coding/assets/50532905/75fdc669-de13-4a39-8eb0-c4908dc39b49) |
|:--:|
| <b> [Figure18] YOLO 튜닝 모델 인식 결과 </b> |
<br>
<br>
iii) Application
<br>
<br>
<b>시연 영상
https://youtu.be/u0yVr2vYNkU?si=l6BmoKHWTctFtOcE </b>
<br>
<br>

## 5. 고도화 및 추후 개선 방안
<br>
<b>교육 콘텐츠</b> 
<br>
- 사용자가 창의성을 발휘해서 문제를 해결할 수 있는 콘텐츠를 개발한다.
<br>
<b>게임성</b>
<br>
- 흥미를 통해 스스로 학습에 몰두할 수 있도록 게임성을 확보해 줄 필요가 있다.
<br>
<b>모델 유연성</b>
<br>
- 다양한 각도에서 찍더라도 100%에 가깝게 인식할 수 있도록 모델의 유연성과 정확도를 올려야 한다.
<br>
<b>지연 시간 최소화</b>
<br>
- 긴 로딩 시간은 사용자의 흥미를 저하시키므로 3초 이내로 작동할 수 있도록 지연 시간을 최소화해야 한다.
<br>
<br>

## 6. 개발 환경
<br>
<b>프로그래밍 언어 및 라이브러리</b>
<br>
- Python, Dart, HTML, Gradio, Yolo, Tensorflow, OpenCV
<br>
<b>통합 개발 환경(IDE)</b>
<br>
- VS Code
<br>
<b>언어 및 프레임워크 연구</b>
<br>
- Beeware, kivy, kotlin, react, Android Studio,  Flutter 
<br>
- Github Copilot 
<br>
<b>프로젝트 관리 도구</b>
<br>
- GitHub, roboflow
<br>
<br>
<b>[yolo모델 학습 개발환경]</b>
<br>
ultralytics (version 8.0.2)<br>
python (version 3.11.5)<br>
gradio (version 4.0.2)<br>
torch (version 2.1.0)<br>
opencv (version 4.8.0)<br>
numpy (version 1.24.3)<br>
<br>
<br>
<b>[AWS EC2 환경]</b>
<br>
Ubuntu 22.04 LTS<br>
python (version 3.10.12)<br>
tensorflow-cpu (version 2.15.0)<br>
opencv-python (version 4.8.1.78)<br>
numpy (version 1.26.2)<br>
torch (version 2.1.0)<br>
ultralytics (version 8.0.209)<br>
firebase-admin (version 6.2.0)<br>
fastapi (version 0.104.1)<br>
uvicorn (version 0.24.0.post1)<br>
<br>
<br>

## 7. 소스 코드
<br>
<b>깃허브 주소</b>
<br>
https://github.com/bigdata4th-first-line/block_coding.git
<br>
<br>
<b>yolo 이미지 인식 모델(Hugging Face)</b>
<br>
https://tongsil-find-arrow-direction-number.hf.space
<br>
<br>

## 8. 참고 문헌
<br>
5세 유아의 특징 - 5세 발달 특성>연령별 발달정보>육아정보 | 아동교육연구지원센터 (silla.ac.kr)<br>
OSMO - https://www.playosmo.com/ko-KR/<br>
Yolov8 이미지 인식 Computer Vision & Machine Intelligence Lab. - YOLOv8 - Computer Vision & Machine Intelligence (catholic.ac.kr)<br>
Opencv - https://github.com/opencv/opencv<br>
OpenCV - Open Computer Vision Library<br>
dynamo db, s3 - https://docs.aws.amazon.com/<br>
