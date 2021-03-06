### 프로젝트 소개
악플 내용을 분석하여 필터링한 후 반환해 주는 웹 서비스입니다.
|Index Page|Predict Page|
|------|---|
|![image](https://user-images.githubusercontent.com/79235021/167577140-1aac12c4-a411-42cf-a36c-5b231f1ac001.png)|![image](https://user-images.githubusercontent.com/79235021/167576805-4a0bccab-6897-4ae5-856c-7e76d59d7ca5.png)|

### 기능 정의
1. / 경로로 접속하면 index 페이지가 출력됩니다.  
    1-1. 인덱스 페이지에서 서비스에 대한 간단한 설명을 보여줍니다.  
    1-2. Filltering 버튼을 통해 /predict API를 요청할 수 있습니다.  
    1-3. Filltering이 종료되면 result 페이지가 출력됩니다.
  
2. /predict API를 요청하여 문장을 보내면 해당 내용을 분석하여 욕설이 필터링되어 반환됩니다.  
    2-1. 사용자로부터 문장을 전달 받습니다.  
    2-2. 전달받은 문장을 .h5 형태로 저장되어 있는 머신러닝 모델에 전달합니다.  
    2-3. 전달된 문장은 필터링되어 사용자에게 반환됩니다.

### Reference
- https://wikidocs.net/81504
- https://html5up.net/
