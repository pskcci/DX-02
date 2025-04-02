## Project ABC

* 프로젝트명 : bolt_detec

```shell
 내용 : 스마트 팩토리 공정중 나사 체결상태의 검토역할을 AI에게 전담한다.
 목표 : 나사 검수에 사용하던 방식을 응용하여 여러가지 공정의 검수 과정을 AI로 대체하여 공정 최적화를 노린다.
```

## High Level Design

* 프로젝트 아키텍쳐

```shell

작성필요     (프로젝트 아키텍쳐 기술, 전반적인 diagram 으로 설명을 권장) 
 ┼아키텍쳐는 차후 추가 예정

```
## Clone code

* 팀 Haidi repository 주소

```shell
 프로젝트 인원은 아래 링크를 이용하여 리포지토리를 로컬로 가져온 다음 양식에 따라 브랜치를 만들고 작업한다.

 브랜치양식 : git check out -b 본인이름/작업날짜/진행항목(예 : bolt_detec)
                                    └ 작업 날짜에 따라 며칠을 연달아 이어서 작업한다면 당일날짜 브런치는 백업용으로
                                    └ 최종 브런치는 '/시작날짜~최종날짜' 의 양식으로 진행
                                      
 링크 : git clone https://github.com/gamdumdum/team-haidi-project.git
```
## Prerequite (가이드 문서)

* 구성과 종속성

```shell
 구성(configuration)
    1. 가상환경 (자세한 사항은 Steps to build의 '가상환경 생성' 항목 참고)
    ┼프로젝트 진행에 따라 독커 환경으로 변동 될 수 있음
```

```shell
종속성(dependencies)
    └ openVINO
    └ AI model yolo(+add learning)
    ┼프로젝트 진행에 따라 추가사항 생성예정
```

## Steps to build

* 절차 (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
 가상환경 생성
    └ python -m venv .venv
    └ source .venv/bin/activate
    └ pip install -U pip
    └ pip install -r requirements.txt
    └ cd ~/xxxx
           └ 가상환경의 위치 or 실행파일 위치
```

```shell
 독커 환경 로드
    ┼프로젝트 진행에 에 따라 추가 작성 예정
```

## Steps to run

* 실행방법

```shell
 논의중 (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

 step1 AI검토파일 실행 순서
    └ cd ~/xxxx
    └ source .venv/bin/activate
    └ cd /path/to/repo/xxx/
    └ python demo.py -i xxx -m yyy -d zzz

 step2 PLC 구동

 step2 기물 준비 및 공정 시작

```

## Output

* 프로젝트 사진

```shell

 ┼사진은 차후추가 예정

![./result.jpg](./result.jpg)

```

## Appendix

```shell
 필요기술
    └ 나사 검수에 사용하기 적절한 AI 모델
    └ 나사 검수의 정확도를 높이기 위한 AI 런닝방법
```

```shell
 알아두어야 할 사항들
    └ AI 딥러닝 효과
    └ 실 공정 현장에서의 모습
    └ 현장 작업자의 의견 및 AI를 이용한 공정 검수에 대한 현장의 의견 등
```

