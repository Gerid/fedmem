from pathlib import Path

import win32com.client as win32


def rgb(r: int, g: int, b: int) -> int:
    return r + (g << 8) + (b << 16)


PP_LAYOUT_BLANK = 12
MSO_TEXT_ORIENTATION_HORIZONTAL = 1
MSO_SHAPE_RECTANGLE = 1
MSO_FALSE = 0
MSO_TRUE = -1
PP_SAVE_AS_OPEN_XML_PRESENTATION = 24
PP_SAVE_AS_PDF = 32


ROOT = Path(r"E:\fedprotrack\paper")
OUT_DIR = ROOT / "presentation"
FIG_DIR = ROOT / "figures"

PPTX_PATH = OUT_DIR / "main_final_teacher_report.pptx"
PDF_PATH = OUT_DIR / "main_final_teacher_report.pdf"

PHASE_PATH = FIG_DIR / "phase_diagram.png"
BEST_PATH = FIG_DIR / "best_method.png"
REGIME_PATH = FIG_DIR / "regime_boundary.png"

SLIDE_WIDTH = 960
SLIDE_HEIGHT = 540

COLORS = {
    "navy": rgb(15, 30, 43),
    "blue": rgb(20, 90, 138),
    "orange": rgb(200, 120, 26),
    "light": rgb(242, 245, 247),
    "panel": rgb(255, 255, 255),
    "soft_blue": rgb(242, 248, 255),
    "text": rgb(42, 42, 42),
    "muted": rgb(107, 114, 128),
    "line": rgb(216, 221, 230),
    "white": rgb(255, 255, 255),
}


def set_shape_text(shape, text, font_size=20, color=None, font_name="Microsoft YaHei", bold=False):
    text_range = shape.TextFrame.TextRange
    text_range.Text = text
    text_range.Font.Name = font_name
    text_range.Font.Size = font_size
    text_range.Font.Bold = int(bool(bold))
    text_range.Font.Color.RGB = COLORS["text"] if color is None else color
    return shape


def add_textbox(slide, text, left, top, width, height, font_size=18, color=None, bold=False):
    box = slide.Shapes.AddTextbox(MSO_TEXT_ORIENTATION_HORIZONTAL, left, top, width, height)
    return set_shape_text(box, text, font_size=font_size, color=color, bold=bold)


def add_shape(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.Shapes.AddShape(MSO_SHAPE_RECTANGLE, left, top, width, height)
    shape.Fill.ForeColor.RGB = fill_color
    if line_color is None:
        shape.Line.Visible = MSO_FALSE
    else:
        shape.Line.ForeColor.RGB = line_color
        shape.Line.Weight = 1.0
    return shape


def add_card(slide, left, top, width, height, fill_color=None):
    return add_shape(
        slide,
        left,
        top,
        width,
        height,
        COLORS["panel"] if fill_color is None else fill_color,
        COLORS["line"],
    )


def add_bullet_box(slide, lines, left, top, width, height, font_size=18):
    box = slide.Shapes.AddTextbox(MSO_TEXT_ORIENTATION_HORIZONTAL, left, top, width, height)
    text_range = box.TextFrame.TextRange
    text_range.Text = "\r\n".join(lines)
    text_range.Font.Name = "Microsoft YaHei"
    text_range.Font.Size = font_size
    text_range.Font.Color.RGB = COLORS["text"]
    text_range.ParagraphFormat.Bullet.Visible = MSO_TRUE
    text_range.ParagraphFormat.Bullet.Character = 8226
    return box


def add_base_layout(slide, section, page_no):
    add_shape(slide, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT, COLORS["light"])
    add_shape(slide, 0, 0, SLIDE_WIDTH, 18, COLORS["blue"])
    add_shape(slide, 0, 18, 18, SLIDE_HEIGHT - 18, COLORS["navy"])
    add_textbox(slide, section, 36, 26, 170, 20, font_size=10, color=COLORS["muted"])
    num = add_textbox(slide, str(page_no), 900, 505, 40, 20, font_size=10, color=COLORS["muted"])
    num.TextFrame.TextRange.ParagraphFormat.Alignment = 3


def add_title(slide, title, subtitle=""):
    add_textbox(slide, title, 36, 56, 860, 56, font_size=24, color=COLORS["text"], bold=True)
    if subtitle:
        add_textbox(slide, subtitle, 36, 108, 860, 28, font_size=12, color=COLORS["muted"])


def add_quote_line(slide, text, top):
    add_shape(slide, 52, top + 4, 6, 28, COLORS["orange"])
    add_textbox(slide, text, 66, top, 820, 36, font_size=16, color=COLORS["navy"], bold=True)


def add_image(slide, path, left, top, width, height):
    slide.Shapes.AddPicture(str(path), MSO_FALSE, MSO_TRUE, left, top, width, height)


def reserve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    try:
        path.unlink()
        return path
    except PermissionError:
        index = 2
        while True:
            candidate = path.with_name(f"{path.stem}_v{index}{path.suffix}")
            if not candidate.exists():
                return candidate
            index += 1


def build_deck():
    powerpoint = win32.Dispatch("PowerPoint.Application")
    powerpoint.Visible = True
    presentation = powerpoint.Presentations.Add()
    presentation.PageSetup.SlideWidth = SLIDE_WIDTH
    presentation.PageSetup.SlideHeight = SLIDE_HEIGHT

    try:
        slide = presentation.Slides.Add(1, PP_LAYOUT_BLANK)
        add_shape(slide, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT, COLORS["navy"])
        add_shape(slide, 0, 0, SLIDE_WIDTH, 22, COLORS["orange"])
        add_textbox(slide, "When Does Concept-Level Aggregation Help?", 54, 96, 780, 52, 26, COLORS["white"], True)
        add_textbox(
            slide,
            "A Sharp Characterization in a Canonical Federated Model",
            54,
            142,
            760,
            28,
            15,
            COLORS["white"],
        )
        add_textbox(slide, "汇报主题：联邦概念漂移下的聚合粒度理论", 54, 212, 760, 34, 20, COLORS["light"], True)
        add_textbox(slide, "主线：不是讲怎么分组，而是讲什么时候值得分组。", 54, 258, 780, 28, 15, COLORS["light"])
        add_textbox(slide, "基于最新版 main final / main.tex 整理", 54, 468, 300, 20, 10, COLORS["light"])
        add_textbox(slide, "第 1 页", 882, 500, 40, 20, 10, COLORS["light"])

        slide = presentation.Slides.Add(2, PP_LAYOUT_BLANK)
        add_base_layout(slide, "问题背景", 2)
        add_title(slide, "联邦概念漂移下，服务端到底该怎么聚合？")
        add_card(slide, 40, 150, 410, 280)
        add_card(slide, 490, 150, 410, 280)
        add_textbox(slide, "策略 A：全局聚合", 62, 176, 220, 28, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "把所有客户端统一合成一个模型",
                "优点：样本多，方差低",
                "缺点：不同概念混在一起会产生干扰偏差",
            ],
            62,
            214,
            350,
            150,
            17,
        )
        add_textbox(slide, "策略 B：概念级聚合", 512, 176, 240, 28, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "按概念组分别建模",
                "优点：避免跨概念干扰",
                "缺点：每组样本少，方差升高",
            ],
            512,
            214,
            340,
            150,
            17,
        )
        add_quote_line(slide, "核心矛盾：global pooling 降低方差，但 concept-level aggregation 降低偏差。", 446)

        slide = presentation.Slides.Add(3, PP_LAYOUT_BLANK)
        add_base_layout(slide, "研究问题", 3)
        add_title(slide, "不是讲怎么分组，而是讲什么时候值得分组")
        add_card(slide, 48, 156, 850, 260)
        add_bullet_box(
            slide,
            [
                "IFCA、CFL 关注 how to cluster：给定要分组，如何恢复正确簇结构",
                "APFL、Ditto 关注 how to interpolate：如何在局部和全局模型间混合",
                "本文追问的是 when to cluster：按概念分组的收益，何时能超过方差代价",
            ],
            72,
            196,
            786,
            150,
            18,
        )
        add_quote_line(slide, "关键点：给出一个明确阈值，判断什么时候该做概念级聚合。", 436)

        slide = presentation.Slides.Add(4, PP_LAYOUT_BLANK)
        add_base_layout(slide, "理论设定", 4)
        add_title(slide, "Canonical 设定：先把最关键的统计矛盾单独剥离出来")
        add_card(slide, 48, 150, 400, 292)
        add_card(slide, 476, 150, 422, 292)
        add_textbox(slide, "问题设定", 72, 176, 150, 26, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "K 个客户端，C 个平衡概念",
                "高斯线性回归，x ~ N(0, I_d)",
                "每个概念有最优参数 w*_j",
                "假设 oracle concept labels 已知",
            ],
            72,
            214,
            330,
            160,
            17,
        )
        add_textbox(slide, "风险分解", 500, 176, 150, 26, 19, COLORS["navy"], True)
        add_textbox(slide, "Global risk = B_j^2 + sigma^2 d / (Kn)", 500, 216, 330, 26, 18, COLORS["text"], True)
        add_textbox(slide, "Concept risk = sigma^2 d / ((K/C)n)", 500, 252, 330, 26, 18, COLORS["text"], True)
        add_bullet_box(
            slide,
            [
                "B_j^2：概念 j 相对全局中心的偏差",
                "全局模型偏差更大，但方差更小",
                "概念模型偏差更小，但方差更大",
            ],
            500,
            298,
            340,
            120,
            16,
        )

        slide = presentation.Slides.Add(5, PP_LAYOUT_BLANK)
        add_base_layout(slide, "主定理", 5)
        add_title(slide, "主结论：概念级聚合是否更优，取决于一个明确阈值")
        add_card(slide, 86, 146, 776, 118, COLORS["soft_blue"])
        add_textbox(slide, "SNR = Kn · B_j^2 / (σ^2 d)", 176, 176, 520, 30, 24, COLORS["navy"], True)
        add_textbox(
            slide,
            "当 SNR > C - 1 时，concept-level aggregation 优于 global aggregation",
            114,
            214,
            700,
            28,
            18,
            COLORS["orange"],
            True,
        )
        add_card(slide, 48, 294, 850, 162)
        add_bullet_box(
            slide,
            [
                "B_j^2 越大：概念分离越明显，混合偏差越大，更该分组",
                "K 和 n 越大：总样本越多，越能支撑按概念建模",
                "σ^2 越大、d 越高：噪声和维度惩罚越强，更倾向全局聚合",
                "C 越大：每组分到的样本越少，概念级方差惩罚越重",
            ],
            72,
            324,
            790,
            110,
            17,
        )

        slide = presentation.Slides.Add(6, PP_LAYOUT_BLANK)
        add_base_layout(slide, "理论贡献", 6)
        add_title(slide, "除了阈值，作者还给了下界证明和一个折中方案")
        add_card(slide, 48, 154, 392, 270)
        add_card(slide, 462, 154, 436, 270)
        add_textbox(slide, "Minimax lower bounds", 70, 182, 240, 28, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "单一全局估计器无法绕开 bias floor",
                "概念级估计器无法绕开 variance floor",
                "说明二者张力是结构性的，不是调参问题",
            ],
            70,
            220,
            330,
            128,
            17,
        )
        add_textbox(slide, "Empirical-Bayes shrinkage", 484, 182, 280, 28, 19, COLORS["navy"], True)
        add_textbox(slide, "w_shrink = (1 - λ) w_j + λ w_bar", 484, 220, 320, 28, 20, COLORS["text"], True)
        add_bullet_box(
            slide,
            [
                "概念差异大时，λ 更小，更接近概念级聚合",
                "概念差异小或样本不足时，λ 更大，更接近全局聚合",
                "λ 由数据自适应估计，无需额外手调",
            ],
            484,
            264,
            350,
            124,
            16,
        )
        add_quote_line(slide, "不只是在做理论判断，也给了一个可操作的估计器。", 440)

        slide = presentation.Slides.Add(7, PP_LAYOUT_BLANK)
        add_base_layout(slide, "合成实验", 7)
        add_title(slide, "合成实验基本把阈值关系跑出来了")
        add_image(slide, PHASE_PATH, 52, 150, 388, 238)
        add_image(slide, BEST_PATH, 462, 150, 388, 238)
        add_card(slide, 52, 404, 798, 70)
        add_bullet_box(
            slide,
            [
                "Theory-experiment alignment = 99/108 = 91.7%，所有 mismatch 都集中在边界附近",
                "Shrinkage 获得 67 次 best-by-MSE，并且在 78/108 = 72.2% 配置中优于两种纯策略",
                "Mean regret = -0.002，worst-case excess < 0.08，说明插值策略稳健",
            ],
            70,
            418,
            760,
            50,
            14,
        )

        slide = presentation.Slides.Add(8, PP_LAYOUT_BLANK)
        add_base_layout(slide, "机制解释", 8)
        add_title(slide, "概念刚切换的几轮，未必适合马上做概念级聚合")
        add_image(slide, REGIME_PATH, 56, 156, 396, 252)
        add_card(slide, 480, 156, 388, 252)
        add_bullet_box(
            slide,
            [
                "长期看 SNR 高，不代表每一轮都应做概念级聚合",
                "刚发生概念切换时，有效样本数 n_eff(s) 还很小",
                "这会把瞬时有效 SNR 拉低到阈值以下",
                "因此在 transient 区间，FedAvg 可能暂时更好",
            ],
            506,
            198,
            332,
            132,
            18,
        )
        add_textbox(slide, "这相当于给出了非平稳场景里的过渡区间。", 506, 352, 326, 44, 16, COLORS["navy"], True)

        slide = presentation.Slides.Add(9, PP_LAYOUT_BLANK)
        add_base_layout(slide, "真实数据桥接", 9)
        add_title(slide, "CIFAR-100 bridge：最后问题落在有效维度估计上")
        add_card(slide, 48, 152, 850, 292)
        add_bullet_box(
            slide,
            [
                "在冻结 ResNet-18 特征上构造 B_j^2 proxy，测试高 SNR 与低 SNR 两侧",
                "高 SNR（disjoint labels）下，Oracle 3/3 seed 全部占优，和理论一致",
                "低 SNR（shared labels）下，若直接用原始维度 d = 128，会得到错误判断",
                "作者检查协方差谱后发现 effective rank 只有约 22，修正后 d_eff^corr ≈ 430",
                "做完 effective-dimension correction 后，CIFAR 两侧一共 6 个 seed 全部与理论方向一致",
            ],
            74,
            188,
            782,
            150,
            17,
        )
        add_quote_line(slide, "作者把这个不一致继续往下追，最后定位到 effective rank。", 438)

        slide = presentation.Slides.Add(10, PP_LAYOUT_BLANK)
        add_base_layout(slide, "评价与局限", 10)
        add_title(slide, "贡献很清楚，但结论主要还是在理想设定里成立")
        add_card(slide, 48, 154, 408, 280)
        add_card(slide, 486, 154, 412, 280)
        add_textbox(slide, "可以抓的三点贡献", 72, 182, 230, 26, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "把问题从“怎么聚类”提升到“何时值得聚类”",
                "给出 sharp threshold 与 matching lower bounds",
                "提出可落地的 shrinkage 自适应策略",
            ],
            72,
            220,
            330,
            126,
            17,
        )
        add_textbox(slide, "需要诚实指出的局限", 510, 182, 240, 26, 19, COLORS["navy"], True)
        add_bullet_box(
            slide,
            [
                "oracle concept labels 假设很强，现实里最难的是概念身份识别",
                "理论依赖高斯、线性、平衡概念、已知协方差等理想条件",
                "effective dimension 修正更像经验 bridge，不是严格推广",
            ],
            510,
            220,
            338,
            146,
            16,
        )
        add_quote_line(slide, "更合适的定位：一篇分析框架很强的理论论文。", 438)

        slide = presentation.Slides.Add(11, PP_LAYOUT_BLANK)
        add_base_layout(slide, "结论", 11)
        add_title(slide, "最后一句话")
        add_card(slide, 76, 164, 788, 162, COLORS["soft_blue"])
        add_textbox(slide, "Concept-level aggregation is useful only when concept separation", 118, 202, 680, 28, 21, COLORS["navy"], True)
        add_textbox(slide, "beats the extra variance cost caused by splitting.", 204, 236, 500, 28, 21, COLORS["navy"], True)
        add_card(slide, 120, 360, 700, 92)
        add_textbox(slide, "一句中文总结：", 148, 386, 150, 24, 17, COLORS["orange"], True)
        add_textbox(slide, "分概念聚合不是默认更好，关键看概念分离收益能不能盖过方差代价。", 148, 416, 640, 28, 17, COLORS["text"])

        slide = presentation.Slides.Add(12, PP_LAYOUT_BLANK)
        add_base_layout(slide, "备答", 12)
        add_title(slide, "老师可能会问的 3 个问题")
        add_card(slide, 48, 154, 850, 292)
        add_bullet_box(
            slide,
            [
                "Q1：和 IFCA、APFL 的区别是什么？ A：本文研究的是聚合粒度决策边界，不是更强聚类算法。",
                "Q2：oracle label 假设太强，是否失去现实意义？ A：它的作用是先剥离概念识别误差，得到一个统计基准。",
                "Q3：CIFAR 的有效维度修正会不会太 ad hoc？ A：这是经验 bridge，说明方向可迁移，但不等于严格理论推广。",
            ],
            74,
            198,
            790,
            178,
            18,
        )
        add_textbox(slide, "答辩时先承认假设强，再把重点拉回“什么时候值得分组”这个问题。", 74, 392, 786, 32, 16, COLORS["navy"], True)

        pptx_out = reserve_output_path(PPTX_PATH)
        pdf_out = reserve_output_path(PDF_PATH)

        presentation.SaveAs(str(pptx_out), PP_SAVE_AS_OPEN_XML_PRESENTATION)
        presentation.SaveAs(str(pdf_out), PP_SAVE_AS_PDF)
    finally:
        try:
            presentation.Close()
        except Exception:
            pass
        try:
            powerpoint.Quit()
        except Exception:
            pass


if __name__ == "__main__":
    build_deck()
