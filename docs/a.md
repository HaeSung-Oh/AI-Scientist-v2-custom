# BFTS를 Codex처럼 도구 루프 기반으로 바꾸는 방법

## 현재 구조의 문제

현재 BFTS agent는 여러 node를 시도하고 실패하면 debug node를 만들기는 하지만, 각 node 내부의 코드 생성은 대체로 다음 구조에 가깝다.

```text
긴 프롬프트 1회
-> 완성된 단일 Python 코드 생성
-> 실행
-> 실패 로그를 다음 프롬프트에 전달
-> 다시 완성 코드 생성
```

즉, "여러 번 시도"는 하지만 Codex처럼 다음 루프를 수행하지는 않는다.

```text
파일 읽기
-> rg로 관련 코드 검색
-> 작은 패치
-> py_compile
-> smoke test
-> 로그 분석
-> 다시 패치
-> git diff 확인
```

그래서 복잡한 연구 코드에서는 모델이 데이터 경로를 상상하거나, 없는 패키지를 import하거나, PyTorch tensor shape을 틀리는 문제가 잘 생긴다.

## 목표 구조

BFTS 자체는 유지하되, "코드 생성기" 부분을 single-shot generator에서 tool-loop code agent로 바꾼다.

```text
BFTS tree search
  -> node 생성 요청
    -> ToolLoopCodeAgent
       -> inspect files
       -> search repo
       -> write/patch code
       -> py_compile
       -> import check
       -> dataset smoke test
       -> short run
       -> final code 반환
  -> BFTS가 기존처럼 실행/평가/선택
```

핵심은 BFTS의 탐색 구조를 버리는 것이 아니라, `plan_and_code_query()` 안쪽을 더 똑똑한 코드 작성 루프로 바꾸는 것이다.

## 1단계: 가장 빠른 현실적 개선

이미 `launch_scientist_bfts.py`에는 `--load_code` 옵션이 있다. 이걸 쓰면 idea JSON과 같은 이름의 `.py` 파일을 starter code로 프롬프트에 넣을 수 있다.

예:

```text
ai_scientist/ideas/my_topic.json
ai_scientist/ideas/my_topic.py
```

실행:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --idea_idx 0 \
  --num_cite_rounds 2 \
  --load_code \
  --code coder
```

이 방식은 완전한 Codex식 tool loop는 아니지만, 최소한 모델이 빈 화면에서 전체 실험 코드를 상상하지 않게 만든다.

추천 starter code 구성:

```text
1. 데이터 경로 확인
2. Dataset/DataLoader
3. baseline model
4. train/eval loop
5. metric 계산
6. experiment_data.npy 저장
7. TODO 영역: 연구 아이디어가 들어갈 부분
```

모델에게는 "전체를 새로 작성"하게 하지 말고, "TODO 영역만 개선"하게 해야 한다.

## 2단계: BFTS 내부에 도구 루프 추가

새 파일을 만든다.

```text
ai_scientist/treesearch/tool_code_agent.py
```

이 agent는 LLM에게 바로 최종 코드를 쓰라고 하지 않고, JSON action을 반복해서 받는다.

예상 action:

```json
{"action": "rg", "pattern": "Dataset", "path": "."}
{"action": "read_file", "path": "ai_scientist/ideas/my_topic.py", "start": 1, "end": 200}
{"action": "write_code", "path": "runfile.py", "content": "..."}
{"action": "run_cmd", "cmd": "python -m py_compile runfile.py"}
{"action": "run_cmd", "cmd": "python smoke_test.py"}
{"action": "final_code", "code": "..."}
```

허용할 tool은 처음에는 작게 시작하는 게 좋다.

```text
read_file
list_files
rg
write_code
run_py_compile
run_import_check
run_smoke_test
git_diff
```

`run_cmd`를 너무 자유롭게 열면 위험하고 재현성도 나빠진다. 처음에는 whitelist 방식이 좋다.

허용 command 예:

```text
python -m py_compile runfile.py
python smoke_test.py
python runfile.py --smoke-test
git diff -- runfile.py
rg <pattern> <path>
```

## 3단계: `parallel_agent.py` 연결 지점

현재 핵심 지점은 다음 함수다.

```text
ai_scientist/treesearch/parallel_agent.py
MinimalAgent.plan_and_code_query()
```

지금은 여기서 LLM query를 한 번 호출하고 코드 블록을 추출한다.

변경 방향:

```python
def plan_and_code_query(self, prompt, retries=3):
    if self.cfg.agent.code.get("mode", "single_shot") == "tool_loop":
        return ToolLoopCodeAgent(self.cfg, prompt).run()
    return self._single_shot_plan_and_code_query(prompt, retries=retries)
```

config에는 옵션을 추가한다.

```yaml
agent:
  code:
    mode: tool_loop
    model: ollama/qwen2.5-coder:32b
    temp: 0.1
    max_tokens: 12000
    tool_loop:
      max_steps: 16
      max_repairs: 4
      allow_network: false
      allowed_commands:
        - py_compile
        - smoke_test
        - git_diff
        - rg
```

## 4단계: 검증 루프

최종 코드로 인정하기 전에 반드시 통과시킨다.

```text
1. Python syntax compile
2. import smoke test
3. dataset path smoke test
4. tiny batch forward pass
5. 1 epoch 또는 1 step mini-run
6. experiment_data.npy 생성 여부 확인
```

실패하면 BFTS node 실행까지 가지 말고, 같은 tool loop 안에서 다시 고치게 한다.

이게 중요하다. 지금 구조는 실패한 코드를 바로 BFTS 실행에 넘기기 때문에 node가 낭비된다.

## 5단계: 멀티 에이전트 구조

중요한 점은 LLM을 한 번 더 부르는 것이 아니라, 역할이 다른 agent들을 반복적으로 호출하는 것이다.

추천 agent 구성:

```text
PlannerAgent
  -> 이번 node에서 무엇을 구현할지 작게 쪼갠다.

RepoInspectorAgent
  -> 로컬 파일, 기존 코드, 데이터 구조, requirements, 로그를 읽는다.

CoderAgent
  -> 실제 코드를 작성하거나 패치한다.

StaticReviewerAgent
  -> 문법, import, shape, 데이터 경로, 금지된 synthetic validation을 검토한다.

RunnerAgent
  -> py_compile, import check, dataset smoke test, short run을 실행한다.

RepairAgent
  -> 실행 로그를 보고 작은 수정안을 만든다.

FinalReviewerAgent
  -> 최종 코드가 연구 목적, metric 저장, 재현성 조건을 만족하는지 확인한다.
```

흐름은 다음처럼 만든다.

```text
BFTS node request
  -> PlannerAgent
  -> RepoInspectorAgent
  -> CoderAgent
  -> StaticReviewerAgent
  -> RunnerAgent
  -> 실패하면 RepairAgent -> RunnerAgent 반복
  -> FinalReviewerAgent
  -> final code 반환
```

이 구조가 좋은 이유:

```text
1. CoderAgent 혼자 모든 판단을 하지 않는다.
2. ReviewerAgent가 "그럴듯하지만 틀린 코드"를 잡는다.
3. RunnerAgent가 실제 실행 결과로 환각을 제거한다.
4. RepairAgent는 전체 코드를 다시 쓰지 않고 작은 수정만 한다.
5. BFTS는 더 이상 깨진 node를 너무 많이 낭비하지 않는다.
```

처음부터 agent를 너무 많이 만들 필요는 없다. 최소 구현은 세 개면 충분하다.

```text
1. CoderAgent
2. ReviewerAgent
3. Runner/RepairAgent
```

실용적인 첫 버전:

```text
step 1: CoderAgent가 코드 초안 작성
step 2: ReviewerAgent가 문제 목록 작성
step 3: CoderAgent가 리뷰 반영
step 4: RunnerAgent가 py_compile + smoke test 실행
step 5: 실패하면 RepairAgent가 로그 기반 패치
step 6: 최대 N회 반복 후 final code 반환
```

의사코드:

```python
def run_tool_loop(prompt, max_rounds=4):
    context = inspect_repo_and_data(prompt)
    plan = planner_agent(prompt, context)
    code = coder_agent(prompt, context, plan)

    for round_idx in range(max_rounds):
        review = reviewer_agent(prompt, context, code)
        code = coder_agent(prompt, context, plan, review=review, previous_code=code)

        result = runner_agent(code)
        if result.ok:
            final_review = final_reviewer_agent(prompt, context, code, result)
            if final_review.ok:
                return code

        code = repair_agent(prompt, context, code, result.error)

    return code
```

여기서 `runner_agent`는 LLM이 아니라 실제 도구 실행이어야 한다. 즉 `py_compile`, import check, smoke test는 말로 판단하지 말고 실제로 실행한다.

## 6단계: GitHub/문서/외부 코드 사용

Codex처럼 필요한 GitHub를 알아서 가져오게 만들 수도는 있다. 다만 기본값으로 열어두면 위험하다.

추천 방식:

```text
기본값: network off
옵션: --allow-code-agent-network
허용 대상: idea/config에 명시된 GitHub URL 또는 allowlist URL만
동작: clone은 read-only cache 디렉터리에만
```

예:

```yaml
agent:
  code:
    tool_loop:
      allow_network: true
      allowed_repos:
        - https://github.com/openai/CLIP
        - https://github.com/DengPingFan/PraNet
```

무작정 "필요한 repo 알아서 찾아와"는 재현성, 보안, 라이선스 측면에서 위험하다. 연구 실험에서는 어떤 외부 코드를 참고했는지 로그에 남겨야 한다.

## 왜 Codex는 되고 BFTS agent는 안 되나

Codex는 모델 하나만이 아니라 다음 요소가 같이 붙어 있다.

```text
도구 사용 권한
파일 시스템 읽기/쓰기
shell command 실행
검색
patch 적용
테스트 실행
로그 관찰
반복 계획 수정
```

반면 현재 BFTS code agent는 긴 프롬프트를 보고 코드를 생성하는 역할에 가깝다. 그래서 같은 모델이라도 Codex식 환경을 주면 훨씬 낫고, single-shot 환경에 넣으면 훨씬 못해 보인다.

## 추천 구현 순서

1. `--load_code`로 연구별 starter code를 먼저 사용한다.
2. `parallel_agent.py`의 `plan_and_code_query()`를 `single_shot`과 `tool_loop` mode로 분리한다.
3. `tool_code_agent.py`를 만들고 read/search/compile/smoke-test 도구부터 붙인다.
4. `CoderAgent`, `ReviewerAgent`, `Runner/RepairAgent` 3개 역할로 최소 멀티 에이전트 루프를 만든다.
5. 최종 코드 반환 전 validation gate를 강제한다.
6. 필요하면 network/GitHub clone tool을 allowlist 기반으로 추가한다.

가장 중요한 원칙:

```text
BFTS는 연구 아이디어 탐색을 맡기고,
실험 코드의 배관과 검증은 도구 루프가 맡게 한다.
```

## 현재 GitHub 최신화 후 구체 계획

2026-05-24 기준 원격 main에는 ideation 단계용 순차 멀티 에이전트가 이미 들어와 있다.

관련 커밋:

```text
111120f Add optional multi-agent ideation mode
7a573bf Strengthen evidence-grounded ideation checks
aa19496 Slow Semantic Scholar request throttle
```

이 변경은 `perform_ideation_temp_free.py`에서 다음 agent schedule을 사용한다.

```text
ExplorerAgent -> SkepticAgent -> ExperimentAgent -> SynthesizerAgent
```

따라서 BFTS code generation도 같은 패턴을 따라가는 것이 좋다.

### 목표

`parallel_agent.py`의 `MinimalAgent.plan_and_code_query()`를 다음 두 mode로 나눈다.

```text
single_shot:
  기존 방식 유지

sequential_multi:
  여러 code agent를 순차 호출하고 실제 검증 도구를 끼워 넣음
```

### 1차 구현 범위

처음부터 GitHub 검색, 웹 검색, 자유 shell 실행까지 열지 않는다. 1차는 로컬 repo와 안전한 검증만 사용한다.

```text
사용할 LLM agent:
1. CodePlannerAgent
2. CodeWriterAgent
3. CodeReviewerAgent
4. CodeRepairAgent

LLM이 아닌 실제 tool:
1. py_compile
2. import check
3. synthetic-only keyword scan
4. optional smoke test
```

### 순차 흐름

```text
BFTS가 node 생성을 요청
  -> CodePlannerAgent
     구현 범위를 작게 정리

  -> CodeWriterAgent
     첫 코드 작성

  -> CodeReviewerAgent
     import, 데이터 경로, metric 저장, synthetic validation, shape 위험 검토

  -> CodeRepairAgent
     리뷰를 반영해 코드 수정

  -> 실제 검증
     python compile
     import check
     synthetic-only scan

  -> 실패 시
     CodeRepairAgent 재호출
     최대 N회 반복

  -> 성공 시
     final code를 BFTS node로 반환
```

### config 예시

```yaml
agent:
  code:
    mode: sequential_multi
    model: ollama/qwen2.5-coder:32b
    temp: 0.1
    max_tokens: 12000
    sequential_multi:
      max_review_rounds: 2
      max_repair_rounds: 4
      run_py_compile: true
      run_import_check: true
      reject_synthetic_only: true
      run_smoke_test: false
```

### 연결할 파일

```text
ai_scientist/treesearch/parallel_agent.py
  - plan_and_code_query()에서 mode 분기

ai_scientist/treesearch/code_multi_agent.py
  - 새로 추가
  - planner/writer/reviewer/repairer 순차 호출

ai_scientist/treesearch/code_validation.py
  - 새로 추가
  - compile/import/synthetic scan 검증 함수

bfts_config.yaml
  - agent.code.mode 옵션 추가
```

### 왜 병렬보다 순차 먼저인가

처음부터 병렬 reviewer를 여러 개 두면 추적이 어려워진다. 현재 실패 원인은 코드 생성 자체가 불안정한 것이므로, 먼저 다음 순차 루프를 안정화하는 것이 낫다.

```text
작성 -> 리뷰 -> 수정 -> 실행 검증 -> 수정
```

이 구조가 안정되면 그 다음에 reviewer만 병렬화한다.

예:

```text
PackageReviewer
DataReviewer
TorchShapeReviewer
MetricReviewer
```

### 1차 완료 기준

다음 조건을 만족하면 1차 성공으로 본다.

```text
1. 기존 single_shot mode는 그대로 동작한다.
2. sequential_multi mode를 켜면 한 node 안에서 LLM이 최소 3회 이상 순차 호출된다.
3. 최종 code 반환 전 py_compile이 반드시 실행된다.
4. py_compile 실패 시 BFTS 실행으로 넘어가지 않고 repair loop가 돈다.
5. synthetic-only로 보이는 코드는 경고 또는 실패 처리된다.
6. 모든 중간 agent 응답과 validation 결과가 로그에 저장된다.
```

## 2차 구현 범위

2차는 순차 멀티 에이전트가 만든 코드를 실제로 짧게 실행해 보는 단계다. 전체 학습을 돌리는 것이 아니라, 생성 코드 안에 smoke-test branch를 강제한다.

생성 코드는 다음 환경 변수를 확인해야 한다.

```python
os.environ.get("AI_SCIENTIST_SMOKE_TEST") == "1"
```

이 모드에서는 다음만 수행하고 종료해야 한다.

```text
1. 필요한 데이터 경로가 있는지 확인
2. 가능하면 실제 데이터에서 작은 batch 하나를 읽음
3. model forward 또는 tiny train step 1회를 수행
4. working/experiment_data.npy를 저장
5. SMOKE_TEST_PASS를 출력
6. full training 전에 종료
```

2차에서 추가된 validation:

```text
run_smoke_test: true
smoke_test_timeout: 60
require_experiment_data: true
```

이제 최종 코드 반환 전 validation은 다음 순서로 돈다.

```text
1. Python syntax compile
2. top-level import availability check
3. synthetic-only marker scan
4. AI_SCIENTIST_SMOKE_TEST branch 존재 확인
5. subprocess로 smoke test 실제 실행
6. working/experiment_data.npy 생성 확인
```

주의할 점:

```text
smoke test는 generated code가 제대로 branch를 구현해야만 빠르게 끝난다.
branch가 없거나 무시되면 validation이 실패하고 RepairAgent가 다시 코드를 고친다.
```

## 3차 구현 범위

3차는 reviewer를 하나의 일반 리뷰어에서 역할별 리뷰어로 쪼개는 단계다. 목적은 한 reviewer가 모든 문제를 대충 훑는 대신, 실패가 자주 나는 영역을 각각 다른 관점으로 보게 만드는 것이다.

기본 reviewer 목록:

```text
PackageReviewer
  - missing imports
  - wrong import paths
  - package/API hallucinations
  - optional dependency guard 누락

DataReviewer
  - invented dataset paths
  - prepared input/ directory 미사용
  - synthetic-only validation
  - image/mask shape, dtype, split 문제

TorchShapeReviewer
  - model input/output tensor shape
  - loss target shape/dtype mismatch
  - device placement
  - DataLoader batch handling
  - tiny smoke-test feasibility

MetricReviewer
  - evaluation metric 누락
  - working/experiment_data.npy 저장 누락
  - AI_SCIENTIST_SMOKE_TEST branch 누락
  - runtime feasibility
```

흐름:

```text
CodeWriterAgent
  -> PackageReviewer
  -> DataReviewer
  -> TorchShapeReviewer
  -> MetricReviewer
  -> feedback 통합
  -> CodeRepairAgent
```

config:

```yaml
agent:
  code:
    sequential_multi:
      reviewers:
        - PackageReviewer
        - DataReviewer
        - TorchShapeReviewer
        - MetricReviewer
```

아직 병렬 reviewer는 아니다. 우선 순차로 호출해서 로그 추적과 실패 원인 분석을 쉽게 유지한다. 이 구조가 안정되면 reviewer 호출만 병렬화할 수 있다.

## 5차 구현 범위

5차는 generated code agent가 Codex처럼 필요한 로컬 정보를 직접 물어보는 제한적 tool-use 단계다. 자유 shell을 주는 것이 아니라, 읽기 중심의 안전한 JSON action만 허용한다.

허용 tool:

```text
list_files
  - repo/workspace 안의 파일 목록 확인

read_file
  - repo/workspace 안의 텍스트 파일 일부 읽기

rg
  - repo/workspace 안에서 ripgrep 검색

inspect_input
  - 현재 workspace/input 구조와 파일 확장자 개수 관찰

inspect_requirements
  - requirements.txt, pyproject.toml, environment.yml 등 관찰

inspect_imports
  - 주요 Python module import 가능 여부 관찰

finish
  - 충분한 context를 모았다고 선언
```

금지:

```text
arbitrary shell
network clone/search
pip install
git push/reset
file write/delete
```

흐름:

```text
ToolUsingCodeAgent
  -> JSON action 요청
  -> 안전 tool 실행
  -> 결과를 transcript에 기록
  -> 최대 max_tool_steps 반복

SequentialCodeMultiAgent
  -> tool transcript를 Planner/Writer/Reviewers/Repairer에게 전달
```

프롬프트에 들어가는 tool context는 다음 원칙을 갖는다.

```text
Observed local tool context for this run.
This is a snapshot, not a permanent guarantee.
If a file or directory is not listed here, do not assume it exists.
```

config:

```yaml
agent:
  code:
    sequential_multi:
      use_tool_loop: true
      tool_loop:
        max_tool_steps: 8
        max_read_chars: 8000
        max_rg_results: 30
        allowed_tools:
          - list_files
          - read_file
          - rg
          - inspect_input
          - inspect_requirements
          - inspect_imports
          - finish
```

이 단계는 아직 "모든 agent가 마음대로 도구를 계속 호출"하는 완전한 Codex 구조는 아니다. 우선 code generation 앞단에서 안전한 로컬 context를 능동적으로 수집하고, 그 transcript를 이후 agent들에게 공유한다.

## 6차 구현 범위

6차는 validation 실패 후에도 tool loop를 다시 돌리는 단계다. 5차는 코드 작성 전에 한 번 context를 모으는 구조였고, 6차는 실패 로그를 보고 다시 관찰한다.

흐름:

```text
generated code validation 실패
  -> validation feedback을 ToolUsingCodeAgent에 전달
  -> inspect_input / inspect_imports / rg / read_file 등으로 원인 재탐색
  -> updated tool context를 RepairAgent에 전달
  -> repaired code 생성
  -> validation 재실행
```

예:

```text
Smoke test failed: FileNotFoundError input/Kvasir-SEG/images
  -> inspect_input
  -> 실제 input 구조 확인
  -> RepairAgent가 경로 수정

ImportError: albumentations
  -> inspect_imports
  -> albumentations missing 확인
  -> RepairAgent가 import guard 또는 torchvision fallback으로 수정
```

config:

```yaml
agent:
  code:
    sequential_multi:
      tool_repair_on_validation_failure: true
```

이 단계부터 repair는 단순히 에러 문자열만 보는 것이 아니라, 실패 후 새로 수집된 로컬 관찰 결과까지 보고 수정한다.

## 7차 구현 범위

7차는 BFTS 본실행에서 실패한 코드의 debug 경로도 tool-aware repair로 바꾸는 단계다.

기존 debug 흐름:

```text
실행 실패 node
  -> _debug(parent_node)
  -> 실패 로그를 프롬프트에 넣고 새 코드 생성
```

7차 흐름:

```text
실행 실패 node
  -> _debug(parent_node)
  -> parent_node.term_out / plot feedback / time feedback 수집
  -> ToolUsingCodeAgent(extra_context=실패 로그) 실행
  -> CodeRepairAgent가 이전 코드 + 실패 로그 + tool context 기반으로 repair
  -> py_compile / import check / smoke test validation
  -> validation 실패 시 tool 재탐색 후 repair 반복
```

config:

```yaml
agent:
  code:
    sequential_multi:
      debug_with_tool_repair: true
```

이 단계의 의미:

```text
초기 코드 생성만 Codex식으로 만드는 것이 아니라,
BFTS 본실행 실패 이후의 debug node도 Codex식으로 repair한다.
```
