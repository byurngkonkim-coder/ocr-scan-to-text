<!-- eg:skip: 조작 절차서 — 주장하는 문서가 아님 -->
# 회사 PC 오타 사전 현행화 (1회)

2026-11-06 부터 사용자 오타 사전 `engine/user_typos.tsv` 가 git 에 올라간다(노트북 사전 기준).
두 PC 가 각자 추가한 줄은 `.gitattributes` 의 `merge=union` 으로 pull 때 자동으로 합쳐진다.

회사 PC 에는 git 밖에서 쓰던 같은 이름의 사전이 있어 **처음 한 번은 그냥 pull 하면 멈춘다**
(`untracked working tree files would be overwritten by merge: engine/user_typos.tsv`).
아래 순서로 한 번만 처리한다. 프로젝트 폴더에서 실행한다.

1. 회사 PC 사전을 옆으로 옮긴다.

   ```bat
   move engine\user_typos.tsv _백업\user_typos_회사PC.tsv
   ```

2. 받는다.

   ```bat
   git pull
   ```

3. 회사 PC 에만 있던 항목을 합친다(등록 때와 같은 안전장치를 거친다).

   ```bat
   engine\venv\Scripts\python engine\tools\merge_user_typos.py _백업\user_typos_회사PC.tsv
   ```

   출력의 `변경:` 줄은 같은 틀린 말을 두 PC 가 다르게 고친 것 — 회사 PC 쪽으로 바뀌었다. 맞는지 본다.
   `거부:` 줄은 안전장치에 걸려 넣지 않은 항목이다.

4. 올린다.

   ```bat
   git add engine/user_typos.tsv
   git commit -m "chore: 회사 PC 오타 사전 항목 합침"
   git push
   ```

끝나면 이후로는 평소처럼 pull·push 만 하면 된다. `_백업\user_typos_회사PC.tsv` 는 확인 후 지워도 된다.

Claude 에게 맡길 때: "docs/회사PC_오타사전_현행화.md 대로 오타 사전 현행화해줘".
