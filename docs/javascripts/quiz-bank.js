// The self-check: three questions per chapter, written from the chapter's own text. The bank is a JSON literal
// (tests/test_site_loop.py parses it), and each chapter's revision is a hash of its questions, so a changed
// question asks to be answered again before the record counts the chapter's 自测 as done.
(() => {
  "use strict";

  const BANK = {
    "00": [
      {
        "id": "00-1",
        "question": "本项目为什么要求每个项目用自己的虚拟环境 .venv？",
        "options": [
          "让 Python 运行得更快",
          "让项目的依赖和系统 Python 互不影响，删项目时连 .venv 一起删掉",
          "没有 .venv 就导入不了 torch",
          "只有激活了环境才能运行脚本"
        ],
        "answer": 1,
        "explanation": "不同项目可能需要不同版本的包，直接装进系统 Python 会互相影响。.venv 是项目自己的工具箱，不需要激活，始终显式调用 .venv\\Scripts\\python.exe 即可。"
      },
      {
        "id": "00-2",
        "question": "在标准 venv 中，哪一组条件最可靠地说明当前解释器确实是项目的虚拟环境？",
        "options": [
          "命令行提示符前面出现 (.venv)",
          "torch.cuda.is_available() 为真",
          "sys.prefix != sys.base_prefix，且解释器路径落在项目的 .venv 内",
          "sys.prefix == sys.base_prefix"
        ],
        "answer": 2,
        "explanation": "sys.base_prefix 是创建虚拟环境所用的基础 Python，sys.prefix 是当前环境根目录；两者不同并且解释器在项目内，比只看提示符前缀可靠。"
      },
      {
        "id": "00-3",
        "question": "实验固定随机种子，是为了什么？",
        "options": [
          "让小实验更容易复现和比较，判断结果变化来自代码还是随机性",
          "保证所有电脑的浮点结果逐位相同",
          "消除所有误差",
          "让训练更快"
        ],
        "answer": 0,
        "explanation": "固定种子不是消除误差，也不保证跨硬件逐位相同；它让每次运行可比，先预测、再运行、一次只改一个变量。"
      }
    ],
    "01": [
      {
        "id": "01-1",
        "question": "稳定 Softmax 先把每个 logit 减去最大值，为什么结果不变？",
        "options": [
          "因为 exp 是线性函数",
          "因为减去最大值后每个概率都变成 1",
          "因为 PyTorch 会自动归一化",
          "所有 logits 同时减去常数 c 时，分子分母同乘 e^{-c}，约掉后概率不变"
        ],
        "answer": 3,
        "explanation": "e^{z_i-c} / Σ e^{z_j-c} = e^{z_i} / Σ e^{z_j}。取 c = max(z) 后最大指数是 exp(0)=1，避免 exp(1000) 溢出。"
      },
      {
        "id": "01-2",
        "question": "语言模型的 logits 形状是 (B,T,V)，Softmax 应沿哪一维计算？",
        "options": [
          "第 0 维 B，对一批样本归一化",
          "最后一维 V，让每个位置对整个词表分配概率",
          "第 1 维 T，对序列位置归一化",
          "哪一维都可以，结果相同"
        ],
        "answer": 1,
        "explanation": "每个 batch、每个时间位置都要在整个词表上分配概率，所以 softmax(logits, dim=-1)，probs.sum(dim=-1) 的每个元素约等于 1。"
      },
      {
        "id": "01-3",
        "question": "为什么 optimizer.zero_grad() 要放在 loss.backward() 之前？",
        "options": [
          "否则 loss 无法反向传播",
          "为了释放显存",
          "PyTorch 默认累加梯度，不先清零，这一步的更新会混入前几步的梯度",
          "为了让学习率生效"
        ],
        "answer": 2,
        "explanation": "基本顺序是 zero_grad -> backward -> step。梯度默认累加，忘记清零会让更新方向被旧梯度污染。"
      }
    ],
    "02": [
      {
        "id": "02-1",
        "question": "Tokenizer 和语言模型的分工是什么？",
        "options": [
          "Tokenizer 决定怎样切分和编号，语言模型学习 token 序列的概率",
          "两者是同一个东西的两种叫法",
          "Tokenizer 学习概率，语言模型负责切分",
          "id 越大的 token 在语言模型里越重要"
        ],
        "answer": 0,
        "explanation": "Tokenizer 是可逆的编号协议，id 本身没有大小语义；模型只处理整数序列，学习的是下一个 token 的概率。"
      },
      {
        "id": "02-2",
        "question": "token 序列 [3, 8, 2, 5] 构造 next-token 训练对，inputs 和 targets 分别是？",
        "options": [
          "inputs = [3, 8, 2, 5]，targets = [3, 8, 2, 5]",
          "inputs = [8, 2, 5]，targets = [3, 8, 2]",
          "inputs = [3, 8]，targets = [2, 5]",
          "inputs = [3, 8, 2]，targets = [8, 2, 5]"
        ],
        "answer": 3,
        "explanation": "输入与目标错开一位：3 -> 8、8 -> 2、2 -> 5。这就是 Transformer 预训练仍在用的 shifted targets。"
      },
      {
        "id": "02-3",
        "question": "Bigram 语言模型估计的是哪个概率？",
        "options": [
          "P(x_{t+1} | x_{t-1})",
          "P(x_{t+1} | x_t)，只看前一个 token",
          "P(x_t | 全部前文)",
          "P(x_t) 的边缘概率"
        ],
        "answer": 1,
        "explanation": "Bigram 只有一步记忆：当前 token 相同时它无法利用更早的上下文，但它已经包含编码、预测、交叉熵和采样这条训练骨架。"
      }
    ],
    "03": [
      {
        "id": "03-1",
        "question": "缩放点积注意力为什么要把 QK^T 除以 sqrt(D_h)？",
        "options": [
          "Softmax 要求输入小于 1",
          "为了减少参数量",
          "点积的波动规模随 D_h 增大，分数过大会让 Softmax 过度尖锐、不利于训练；缩放控制统计尺度",
          "让结果看起来更小更漂亮"
        ],
        "answer": 2,
        "explanation": "若 q、k 分量大致独立、方差 1，长度 D_h 的点积波动会随 D_h 增大；除以 sqrt(D_h) 把分数拉回稳定尺度。"
      },
      {
        "id": "03-2",
        "question": "加了因果掩码后，位置 i 能读取哪些位置的 token？",
        "options": [
          "j <= i，自己和过去",
          "j >= i，自己和未来",
          "所有位置",
          "只有 i 自己"
        ],
        "answer": 0,
        "explanation": "一次输入完整序列是为了并行训练；为了不偷看答案，M_ij 在 j <= i 时为 0，j > i 时为 -inf。"
      },
      {
        "id": "03-3",
        "question": "本章说“最重要的无泄漏测试”是哪一种？",
        "options": [
          "目测 mask 图案是下三角",
          "看训练 loss 是否下降",
          "检查每行注意力权重之和为 1",
          "两条相同序列只改第二条的最后一个 token，在 eval() 和 no_grad() 下前缀输出应在浮点误差内相同"
        ],
        "answer": 3,
        "explanation": "行为测试比看图案更强：allclose(output_a[:, :-1], output_b[:, :-1]) 应为真；前缀也变化说明 mask 方向、广播或维度有错。"
      }
    ],
    "04": [
      {
        "id": "04-1",
        "question": "Decoder-only 模型（GPT、本项目的 TinyGPT）的可见信息和训练目标是？",
        "options": [
          "双向可见，恢复被遮的 [MASK] 位置（MLM）",
          "整个序列用因果 mask，预测下一个 token（CLM）",
          "Encoder 双向读源序列，Decoder 通过 cross-attention 读取它",
          "没有 mask，直接输出整句"
        ],
        "answer": 1,
        "explanation": "x1 -> 预测 x2，x1 x2 -> 预测 x3……统一的 next-token 接口让任务很容易表示为“给定提示，继续生成”。"
      },
      {
        "id": "04-2",
        "question": "SFT 把用户和格式 token 的 label 设为 -100，是为了什么？",
        "options": [
          "把这些 token 从输入里删掉",
          "让模型忘记提示内容",
          "模型仍接收完整提示，但只在助手回答位置计算交叉熵",
          "加快 Tokenizer 的速度"
        ],
        "answer": 2,
        "explanation": "监督信号集中在要学的回答上；提示仍是可见上下文。多轮对话时监督哪几轮必须明确，不能硬编码某个 token id。"
      },
      {
        "id": "04-3",
        "question": "长度为 6 的 CLM 序列能形成多少个 next-token 监督位置？",
        "options": [
          "6 个",
          "36 个",
          "1 个",
          "5 个"
        ],
        "answer": 3,
        "explanation": "x = window[:-1]，y = window[1:]：6 个 token 错开一位，得到 5 对输入和目标。"
      }
    ],
    "05": [
      {
        "id": "05-1",
        "question": "训练前的第一项预测：初始模型接近均匀猜测时，loss 大约是多少？",
        "options": [
          "ln(vocab_size)",
          "0",
          "vocab_size",
          "1"
        ],
        "answer": 0,
        "explanation": "均匀预测时正确 token 的概率是 1/V，-ln(1/V) = ln(V)。这是写进实验记录、再运行对照的数。"
      },
      {
        "id": "05-2",
        "question": "checkpoint 为什么要和模型配置、Tokenizer 一起冻结，而不是只存权重？",
        "options": [
          "文件更大更安全",
          "PyTorch 要求这样保存",
          "为了以后恢复训练",
          "只有权重往往无法正确重建模型；Tokenizer 的 id 映射一变，即使 tensor 形状相同语义也完全错位"
        ],
        "answer": 3,
        "explanation": "base.pt 带着 model_state、配置、冻结的字符 Tokenizer、随机种子和语料指纹；实验 07 和 10 的严格 loader 校验失败就停止，不部分加载。"
      },
      {
        "id": "05-3",
        "question": "train loss 下降而 val loss 上升，说明什么？",
        "options": [
          "学习率太低",
          "过拟合信号",
          "训练完全成功",
          "数据泄漏"
        ],
        "answer": 1,
        "explanation": "语料前 90% 训练、后 10% 验证：train 降表示越来越适合见过的窗口，val 升表示对留出文本没有改善。"
      }
    ],
    "06": [
      {
        "id": "06-1",
        "question": "LoRA 把线性层改为 h = W0 x + (α/r) B A x，W0 形状 [d_out, d_in]，可训练参数量是？",
        "options": [
          "d_in × d_out",
          "r²",
          "r (d_in + d_out)",
          "d_in + d_out"
        ],
        "answer": 2,
        "explanation": "A 形状 [r, d_in]，B 形状 [d_out, r]；r 远小于输入输出维度时，可训练参数显著少于完整权重的 d_in × d_out。"
      },
      {
        "id": "06-2",
        "question": "本项目为什么把 LoRA 的 B 初始化为 0？",
        "options": [
          "训练开始时 BA = 0，LoRA 层输出与原层完全一致，不会一包裹就破坏预训练函数",
          "让 A 的梯度更大",
          "为了节省内存",
          "因为 B 不参与训练"
        ],
        "answer": 0,
        "explanation": "第一步 A 的梯度可能因 B = 0 而为 0，这是正常现象；B 更新后 A 也能获得梯度。"
      },
      {
        "id": "06-3",
        "question": "实验 10 的 train/dev/test 协议里，学习者可以依据哪个切分调步数、学习率和 rank？",
        "options": [
          "test",
          "train",
          "三个都可以",
          "dev"
        ],
        "answer": 3,
        "explanation": "train 是唯一参与梯度更新的数据，dev 用于调配置，test 只在配置确定后评测一次；不能反复查看 test 结果再选配置。"
      }
    ],
    "07": [
      {
        "id": "07-1",
        "question": "平均负对数似然为 NLL 时，困惑度 PPL 等于？",
        "options": [
          "NLL²",
          "exp(NLL)",
          "log(NLL)",
          "1 / NLL"
        ],
        "answer": 1,
        "explanation": "PPL 表示模型每一步像是在多少个等概率候选中犹豫；不同 Tokenizer 改变序列长度和词表，困惑度不能不加说明地横向比较。"
      },
      {
        "id": "07-2",
        "question": "RAG 失败矩阵里“检索错、生成对”应该怎样诊断？",
        "options": [
          "正常成功",
          "生成未忠实使用证据",
          "可能靠参数记忆猜中，不能算有证据的成功",
          "先改索引、切块、查询和检索器"
        ],
        "answer": 2,
        "explanation": "必须保存 top-k 片段与分数才能做这个分类；没有证据支持的正确答案不算 RAG 成功。"
      },
      {
        "id": "07-3",
        "question": "最小工具循环里，模型提出结构化工具调用之后，下一步是什么？",
        "options": [
          "直接执行模型写出的调用",
          "让模型自己执行",
          "跳过校验，把结果写入回答",
          "程序校验工具名和参数，再在权限边界内执行"
        ],
        "answer": 3,
        "explanation": "Agent 是系统，不是单个提示词：工具、状态、控制循环和停止条件由程序显式控制，未知工具和工具异常都要变成可观察状态。"
      }
    ],
    "08": [
      {
        "id": "08-1",
        "question": "必须提交的目录里，results.csv 的要求是什么？",
        "options": [
          "原始指标，不只截图",
          "一张训练曲线截图",
          "可选，有总分即可",
          "只写结论"
        ],
        "answer": 0,
        "explanation": "report/ 里要有 README、config、results.csv、examples.md 和 limitations.md；代码从项目根目录执行，不硬编码本机绝对路径。"
      },
      {
        "id": "08-2",
        "question": "方向 B 的警告：移除 causal mask 的模型训练 loss 更低，说明什么？",
        "options": [
          "它是更好的语言模型",
          "它偷看了答案，不是更好的语言模型",
          "应该默认移除 mask",
          "训练失败了"
        ],
        "answer": 1,
        "explanation": "没有因果掩码的位置能读到未来 token，loss 低是作弊的结果；正确实现要配未来泄漏测试。"
      },
      {
        "id": "08-3",
        "question": "完成定义要求明确区分哪两件事？",
        "options": [
          "训练集和验证集",
          "Encoder 和 Decoder",
          "“代码管线运行成功”和“模型能力足够好”",
          "CPU 和 GPU"
        ],
        "answer": 2,
        "explanation": "一条命令跑快速测试、另一条重现主要结果、至少一个结论有自动检查支持、至少一个原假设被数据修正，再加上这条区分。"
      }
    ]
  };

  // FNV-1a over the chapter's questions: the record stores it beside quizPassedAt.
  function hash(text) {
    let value = 0x811c9dc5;
    for (let index = 0; index < text.length; index += 1) {
      value ^= text.charCodeAt(index);
      value = Math.imul(value, 0x01000193) >>> 0;
    }
    return value.toString(16).padStart(8, "0");
  }

  const revisions = Object.fromEntries(Object.entries(BANK).map(([id, questions]) => [id, hash(JSON.stringify(questions))]));

  window.LlmQuizBank = Object.freeze({
    chapters: Object.freeze(Object.keys(BANK)),
    questionsFor: (id) => (BANK[id] ?? []).map((question) => ({ ...question, options: question.options.slice() })),
    revision: (id) => revisions[id] ?? null,
    count: (id) => (BANK[id] ?? []).length,
  });
})();
