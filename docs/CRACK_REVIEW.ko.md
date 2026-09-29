# 크랙 후보 검토 초안

러버트랙에서 **크랙**은 고무 표면이나 트레드 밑동에 생긴 선형 균열로 취급합니다. 브리지스톤 사용 설명서는 트레드 밑동의 작은 오존 균열과 굽힘 피로로 인한 밑동 균열을 구분하며, 철심·케이블 노출 여부는 별도로 확인하도록 합니다. [브리지스톤 트랙 설명서](https://www.bridgestone.com/products/diversified/rubbertrack/pdf/Tracks-Operating-Manual-USpdf.pdf)

**청크(chunking)**는 고무 덩어리가 표면에서 뜯겨 나간 상태이고, **칩(chipping)**도 표면 조각 손실을 설명할 때 사용됩니다. 제조사 자료는 두 표현을 함께 다루기도 하므로 두 용어 사이에 공통으로 인정되는 크기 경계가 있다고 가정하지 않습니다. 두 현상은 선형 균열과 같은 영상 객체로 단정하지 않습니다. 이 버전은 선형 크랙 후보만 제안하고 청크·칩 분류는 하지 않습니다. [Trackman 용어집](https://www.cisinc-usa.com/wp-content/uploads/Rubber_Track_Warranty_Guide.pdf), [브리지스톤 손상 사례](https://www.bridgestone.com/products/diversified/rubbertrack/pdf/Tracks-Operating-Manual-USpdf.pdf)

## 탐색과 검토

1. 전개 사진을 영역 마스크의 작업 해상도로 축소하고, 두 크기의 black-hat 필터로 국소적으로 어두운 선을 찾습니다.
2. 사진에서 측정한 피치 기준점을 사용해 두 피치 패턴의 중앙값을 만들고, 같은 위치에서 반복되는 어두운 형상 경계를 억제합니다.
3. 연결된 선형 성분을 추출해 다각형, 영역 ID, 영상 점수, 추정 길이를 만듭니다. 면적·깊이·철심 노출 여부는 계산하지 않습니다.
4. 모든 자동 후보는 **미검토**로 시작합니다. 사용자는 사진과 확대 보기를 보고 **채택·제외·보류**를 선택합니다. 탐색이 놓친 균열은 **수동 균열 그리기**에서 점을 따라 경로를 지정해 채택 상태로 추가할 수 있습니다. 채택된 후보만 확정 개수와 추정 길이에 포함됩니다.

민감도는 낮음/보통/높음 세 단계입니다. 다시 탐색하면 기존 채택·제외 판단과 수동 경로가 초기화됩니다. 영역 지도가 변경되면 결과가 오래된 것으로 표시되며, 재탐색 전에는 검토 상태를 바꿀 수 없습니다. 결과는 작업의 `result/cracks/review.json`에 저장되고 작업 삭제 시 함께 지워집니다.

이 방법은 표면 색과 그림자에 민감합니다. 골부 경계, 스프라켓 홀, 트랙 바깥 경계, 숫자 표기, 찢긴 고무가 잘못 검출될 수 있고 밝거나 얇은 균열은 누락될 수 있습니다. 표시된 길이는 영상의 선형 성분에서 구한 **대략적인 길이**이며, 제품 안전 판정이나 균열 깊이 측정이 아닙니다. 특히 철심 노출 여부는 사진만으로 자동 확정하지 않습니다.

## API

- `GET /api/jobs/{id}/cracks`: 저장된 후보와 검토 상태, 영역별 집계 및 영역 지도 변경 여부.
- `POST /api/jobs/{id}/cracks/propose`: `{"group_id":"geometry","sensitivity":"normal"}`로 새 후보를 생성합니다.
- `PATCH /api/jobs/{id}/cracks/{candidate_id}`: `{"status":"accepted"}`처럼 `pending`·`accepted`·`excluded` 중 하나로 검토합니다.
- `POST /api/jobs/{id}/cracks/manual`: `{"points":[[x,y],[x,y],...]}`로 놓친 균열 경로를 추가합니다.
- `DELETE /api/jobs/{id}/cracks/{candidate_id}`: 수동으로 그린 균열 경로를 삭제합니다.
