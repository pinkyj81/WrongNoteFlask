# WrongNoteFlask

오답노트 모바일 스타일 화면을 Flask + MSSQL(ms1901)로 구현한 예제입니다.

## 1) 설치

```bash
cd WrongNoteFlask
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) DB 접속 환경변수 설정 (ms1901)

앱 폴더에 환경변수 전용 파일을 사용합니다.

1) 예시 파일 복사

```bash
copy .env.example .env
```

2) `.env` 파일 값 입력

```env
DB_SERVER=ms1901.gabiadb.com
DB_NAME=yujincast
DB_USER=여기에_계정
DB_PASSWORD=여기에_비밀번호
DB_DRIVER=ODBC Driver 18 for SQL Server
```

원하면 `.env` 에서 한 줄로 `DB_URI`를 직접 지정할 수 있습니다.

```env
DB_URI=mssql+pyodbc:///?odbc_connect=...
```

환경변수 로딩은 `config.py`에서 자동 처리됩니다.

## 3) 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:5003` 접속.

## 4) 샘플 데이터 넣기

```bash
curl -X POST http://127.0.0.1:5003/seed-sample
```

## API

- `GET /api/notes`: 오답 목록 조회
- `POST /api/notes`: 오답 등록
- `POST /api/upload-image`: 사진 파일 업로드 (multipart/form-data, field name: image)
- `POST /api/init-db`: ms1901/yujincast에 `wrong_notes` 테이블 생성(없으면 생성)
- `POST /seed-sample`: 샘플 데이터 등록
- `GET /health`: 헬스체크

## 새 오답 사진 업로드

- 우측 상단 `+ 새 오답` 버튼 클릭
- 모바일: 카메라 촬영 또는 갤러리 선택
- PC: 파일 선택창에서 이미지 선택
- `텍스트 붙여넣기`에 문제 텍스트를 넣고 `붙여넣기 텍스트 파싱`으로 제목/문제/보기 자동 분리 가능
- 파싱/수동 입력한 문제는 DB에 `question_text`(문제 본문), `choices_json`(보기 배열)로 분리 저장되어 상세 화면에서 별도 로드
- 비슷한 유형은 문제 등록 후 상세 화면의 `비슷한 유형 등록` 버튼으로 팝업에서 따로 추가 가능
- 비슷한 유형 텍스트에 `정답` 표를 함께 붙여넣으면 번호별 정답도 별도 파싱되어 저장됨
- 상세 화면의 비슷한 유형 섹션은 기본 접힘 상태이며 필요할 때만 펼쳐서 확인
- 등록하기 누르면 이미지 업로드 후 오답이 DB에 저장됨

## DB 테이블 생성 확인

환경변수를 ms1901/yujincast로 맞춘 뒤 아래 호출:

```bash
curl -X POST http://127.0.0.1:5003/api/init-db
```

성공 시 `wrong_notes` 테이블이 생성되거나(최초) 기존 테이블을 그대로 사용합니다.
