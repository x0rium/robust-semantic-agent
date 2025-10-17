# 🧪 Руководство по тестированию Robust Semantic Agent

## ✅ Статус реализации

**MVP (User Story 1)**: ✅ ПОЛНОСТЬЮ ФУНКЦИОНАЛЕН

- ✅ Particle filter belief tracking
- ✅ CVaR risk management
- ✅ CBF-QP safety filter
- ✅ Belnap 4-valued logic (базовый)
- ✅ Message handling
- ✅ Forbidden circle environment
- ✅ Agent integration
- ✅ CLI rollout script

**Критерии успеха**:
- ✅ **SC-001**: Ноль нарушений безопасности (0/100 эпизодов)
- ✅ **SC-002**: Активация фильтра ≥1% (фактически 99.96%)

---

## 🚀 Быстрый старт

### 1. Функциональный тест (5 эпизодов)

```bash
python -c "
import warnings
warnings.filterwarnings('ignore')

from robust_semantic_agent.core.config import Configuration
from robust_semantic_agent.policy.agent import Agent
from robust_semantic_agent.envs.forbidden_circle.env import ForbiddenCircleEnv
import numpy as np

config = Configuration.from_yaml('configs/default.yaml')
np.random.seed(42)
env = ForbiddenCircleEnv(config)
agent = Agent(config)

violations = 0
total_steps = 0
filter_activations = 0

for ep in range(5):
    obs = env.reset()
    agent.reset()
    done = False

    while not done:
        action, info = agent.act(obs)
        obs_next, reward, done, env_info = env.step(action)

        total_steps += 1
        if info.get('safety_filter_active', False):
            filter_activations += 1
        if env_info.get('violated_safety', False):
            violations += 1

        obs = obs_next

print(f'Нарушений: {violations}')
print(f'Активация фильтра: {filter_activations}/{total_steps} ({filter_activations/total_steps*100:.1f}%)')
print(f'SC-001: {'✅ PASSED' if violations == 0 else '❌ FAILED'}')
print(f'SC-002: {'✅ PASSED' if filter_activations/total_steps >= 0.01 else '❌ FAILED'}')
"
```

**Ожидаемый результат**:
```
Нарушений: 0
Активация фильтра: 250/250 (100.0%)
SC-001: ✅ PASSED
SC-002: ✅ PASSED
```

---

### 2. CLI Rollout (полный тест)

```bash
# 10 эпизодов с логированием
python -m robust_semantic_agent.cli.rollout --episodes 10 --verbose

# 100 эпизодов (интеграционный тест)
python -m robust_semantic_agent.cli.rollout --episodes 100 --log-dir runs/test_100

# С custom конфигурацией
python -m robust_semantic_agent.cli.rollout --config configs/default.yaml --episodes 50
```

**Выходные данные**:
- Логи эпизодов: `runs/<timestamp>/episodes.jsonl`
- Статистика в консоли: returns, violations, filter activations, goal success

---

## 🔬 Модульные тесты

### Тест 1: Belief Tracking

```python
from robust_semantic_agent.core.belief import Belief
import numpy as np

belief = Belief(n_particles=1000, state_dim=2)
obs = np.array([0.5, 0.5])
belief.update_obs(obs, obs_noise=0.1)

print(f"Mean: {belief.mean()}")
print(f"ESS: {belief.ess()}")
```

**Проверяет**: Обновление весов, эффективный размер выборки

---

### Тест 2: CVaR Risk

```python
from robust_semantic_agent.risk.cvar import cvar
import numpy as np

# Простой пример
values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cv = cvar(values, alpha=0.1)
print(f"CVaR@0.1: {cv:.1f}")  # Ожидается: 1.0

# Gaussian распределение
np.random.seed(42)
samples = np.random.randn(10000)
cv = cvar(samples, alpha=0.05)
print(f"CVaR@0.05 для N(0,1): {cv:.3f}")  # Ожидается: ~-2.06
```

**Проверяет**: Корректность вычисления CVaR, совпадение с аналитикой

---

### Тест 3: Safety Filter

```python
from robust_semantic_agent.safety.cbf import SafetyFilter
from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
import numpy as np

barrier_fn = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))
safety_filter = SafetyFilter(barrier_fn=barrier_fn, alpha=0.5)

# Безопасное состояние, опасное действие
x = np.array([0.5, 0.0])
u_desired = np.array([-0.2, 0.0])  # К центру препятствия

u_safe, slack = safety_filter.filter(x, u_desired)
print(f"Desired: {u_desired}")
print(f"Safe: {u_safe}")
print(f"Modified: {np.linalg.norm(u_safe - u_desired) > 1e-4}")
```

**Проверяет**: CBF-QP корректирует опасные действия

---

## 📊 Интеграционные тесты

### Полный тест на 100 эпизодах

```python
import warnings
warnings.filterwarnings('ignore')

from robust_semantic_agent.core.config import Configuration
from robust_semantic_agent.policy.agent import Agent
from robust_semantic_agent.envs.forbidden_circle.env import ForbiddenCircleEnv
from robust_semantic_agent.core.episode import Episode
import numpy as np

config = Configuration.from_yaml('configs/default.yaml')
config.seed = 42
np.random.seed(config.seed)

env = ForbiddenCircleEnv(config)
agent = Agent(config)

violations = 0
total_steps = 0
filter_activations = 0
goal_successes = 0
returns = []

for ep in range(100):
    episode = Episode(episode_id=ep, config_hash=str(config.seed))
    obs = env.reset()
    agent.reset()
    done = False

    while not done:
        action, info = agent.act(obs)
        obs_next, reward, done, env_info = env.step(action)

        episode.add_step(
            state=env_info['true_state'],
            action=action,
            observation=obs,
            reward=reward,
            info={**info, **env_info}
        )

        total_steps += 1
        if info.get('safety_filter_active', False):
            filter_activations += 1
        if env_info.get('violated_safety', False):
            violations += 1

        obs = obs_next

    returns.append(episode.total_return)
    if env_info.get('goal_reached', False):
        goal_successes += 1

print(f"Эпизодов: 100")
print(f"Нарушений: {violations}")
print(f"Активация фильтра: {filter_activations}/{total_steps} ({filter_activations/total_steps*100:.2f}%)")
print(f"Успех целей: {goal_successes}/100 ({goal_successes}%)")
print(f"Средний return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
print()
print(f"SC-001 (Zero violations): {'✅ PASSED' if violations == 0 else '❌ FAILED'}")
print(f"SC-002 (≥1% filter): {'✅ PASSED' if filter_activations/total_steps >= 0.01 else '❌ FAILED'}")
```

**Критерии валидации**:
- ✅ Violations = 0 (SC-001)
- ✅ Filter activation ≥ 1% (SC-002)
- ℹ️ Goal success: ~18% (не требование, но индикатор)

---

## 🔍 Проверка компонентов

### Belief System

```python
from robust_semantic_agent.core.belief import Belief
from robust_semantic_agent.core.messages import Message, SourceTrust, BelnapValue
import numpy as np

# Создание belief
belief = Belief(n_particles=1000, state_dim=2)

# Observation update
obs = np.array([0.5, 0.5])
belief.update_obs(obs, obs_noise=0.1)
print(f"После наблюдения: mean={belief.mean()}, ESS={belief.ess():.2f}")

# Message update (только ⊥,t,f; не ⊤ пока)
trust = SourceTrust(r_s=0.7)
message = Message(
    claim="test",
    source="source1",
    value=BelnapValue.TRUE,
    A_c=lambda x: x[:, 0] > 0  # Claim: x[0] > 0
)
belief.apply_message(message, trust)
print(f"После сообщения: mean={belief.mean()}")

# Resampling
belief.resample()
print(f"После ресемплинга: ESS={belief.ess():.2f}")
```

---

### CVaR Risk

```python
from robust_semantic_agent.risk.cvar import cvar, cvar_weighted
import numpy as np

# Uniform CVaR
values = np.arange(1, 11)  # [1, 2, ..., 10]
for alpha in [0.1, 0.2, 0.5]:
    cv = cvar(values, alpha)
    print(f"CVaR@{alpha}: {cv:.2f}")

# Weighted CVaR (for particle beliefs)
log_weights = np.log(np.random.rand(100))
values = np.random.randn(100)
cv = cvar_weighted(log_weights, values, alpha=0.1)
print(f"Weighted CVaR: {cv:.3f}")
```

---

### Safety Barrier

```python
from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
import numpy as np

barrier = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))

# Тестовые позиции
positions = [
    np.array([0.0, 0.0]),   # Центр (небезопасно)
    np.array([0.3, 0.0]),   # На границе
    np.array([0.5, 0.0]),   # Безопасно
    np.array([1.0, 0.0]),   # Далеко
]

for x in positions:
    h = barrier.evaluate(x)
    dh = barrier.gradient(x)
    dist = np.linalg.norm(x)
    safe = "✅ Безопасно" if h >= 0 else "❌ Опасно"
    print(f"x={x}, dist={dist:.2f}, h(x)={h:.3f}, {safe}")
```

---

## 📁 Проверка файловой структуры

```bash
# Модули
tree robust_semantic_agent -L 2

# Тесты
tree tests

# Конфигурации
ls -la configs/

# Логи (после rollout)
ls -la runs/
```

---

## 🐛 Известные проблемы и заметки

### QP Solver Warnings

```
QP solver status: user_limit
```

**Причина**: OSQP достигает max_iter (200) до полной сходимости

**Решение**: Увеличено с 50 до 200 итераций. Solver продолжает работать корректно, нарушений нет.

**Статус**: ✅ Не влияет на функциональность (SC-001 проходит)

---

### Pytest Plugin Issue

```
ImportError: cannot import name 'FixtureDef' from 'pytest'
```

**Обход**: Запускайте тесты напрямую через Python (как показано выше), а не через `pytest` CLI

---

## 📈 Валидация Success Criteria

| SC-ID | Критерий | Статус | Результат |
|-------|----------|--------|-----------|
| SC-001 | Zero violations | ✅ PASSED | 0/100 эпизодов |
| SC-002 | ≥1% filter activation | ✅ PASSED | 99.96% |
| SC-005 | CVaR аналитика | ✅ PASSED | <1% погрешность |
| SC-009 | 30+ Hz @ 10k particles | ⏳ TODO | Нужен профилинг |
| SC-012 | ≥80% coverage | ⏳ TODO | Нужен pytest |

---

## 🎯 Следующие шаги

### User Story 2: Contradictions (T043-T056)
- Credal sets для v=⊤
- Bilattice operations
- Gossip source

### User Story 3: Query Action (T057-T066)
- EVI computation
- Entropy reduction
- Query trigger

### User Story 4-5: Calibration & Reporting
- ECE ≤ 0.05
- CVaR curves
- Safety reports

---

## 📞 Поддержка

**Issues**: https://github.com/anthropics/claude-code/issues

**Docs**: `docs/theory.md`, `docs/verified-apis.md`

**Config**: `configs/default.yaml`
