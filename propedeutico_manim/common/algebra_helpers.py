from manimlib import *


TITLE_SIZE = 38
SUBTITLE_SIZE = 24
LABEL_SIZE = 18
BODY_SIZE = 25
EQUATION_SIZE = 34

PANEL_STROKE = GREY_B
EXERCISE_COLOR = BLUE_B
OBJECTIVE_COLOR = TEAL_B
STEP_COLOR = YELLOW
RESULT_COLOR = GREEN_B


def fit_to_width(mobject, max_width):
    if mobject.get_width() > max_width:
        mobject.set_width(max_width)
    return mobject


def scene_title(title, subtitle=None):
    title_mob = Text(title, font_size=TITLE_SIZE)
    fit_to_width(title_mob, 12.7)
    title_mob.to_edge(UP, buff=0.32)

    underline = Line(LEFT, RIGHT, color=EXERCISE_COLOR, stroke_width=2)
    underline.set_width(min(title_mob.get_width(), 11.7))
    underline.next_to(title_mob, DOWN, buff=0.08)

    group = VGroup(title_mob, underline)
    if subtitle:
        subtitle_mob = Text(subtitle, font_size=SUBTITLE_SIZE, color=GREY_A)
        fit_to_width(subtitle_mob, 12.2)
        subtitle_mob.next_to(underline, DOWN, buff=0.12)
        group.add(subtitle_mob)

    return group


def _panel(label, body_mob, width, stroke_color=PANEL_STROKE, label_color=GREY_A):
    label_mob = Text(label.upper(), font_size=LABEL_SIZE, color=label_color)
    fit_to_width(body_mob, width - 0.5)

    content = VGroup(label_mob, body_mob)
    content.arrange(DOWN, aligned_edge=LEFT, buff=0.12)

    box = SurroundingRectangle(content, buff=0.22)
    box.set_stroke(stroke_color, width=1.4)
    box.set_fill(stroke_color, opacity=0.06)

    return VGroup(box, content)


def tex_panel(label, tex, width=6.2, font_size=BODY_SIZE, color=WHITE, stroke_color=PANEL_STROKE):
    body = Tex(tex, font_size=font_size, color=color)
    return _panel(label, body, width, stroke_color=stroke_color, label_color=stroke_color)


def text_panel(label, text, width=5.4, font_size=22, color=WHITE, stroke_color=PANEL_STROKE):
    body = Text(text, font_size=font_size, color=color)
    return _panel(label, body, width, stroke_color=stroke_color, label_color=stroke_color)


def exercise_block(tex, label="Ejercicio", width=6.2, font_size=BODY_SIZE):
    return tex_panel(label, tex, width=width, font_size=font_size, stroke_color=EXERCISE_COLOR)


def exercise_text_block(text, label="Problema", width=6.6, font_size=21):
    return text_panel(label, text, width=width, font_size=font_size, stroke_color=EXERCISE_COLOR)


def objective_block(text, width=5.4):
    return text_panel("Objetivo", text, width=width, font_size=21, stroke_color=OBJECTIVE_COLOR)


def equation_column(lines, font_size=EQUATION_SIZE, max_width=8.4, buff=0.26):
    equations = VGroup()
    for line in lines:
        equation = Tex(line, font_size=font_size)
        fit_to_width(equation, max_width)
        equations.add(equation)
    equations.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return equations


def highlight_step(mobject, color=STEP_COLOR):
    box = SurroundingRectangle(mobject, buff=0.12)
    box.set_stroke(color, width=2)
    box.set_fill(color, opacity=0.06)
    return box


def result_box(tex, label="Resultado", width=11.4, font_size=34):
    label_mob = Text(label, font_size=22, color=RESULT_COLOR)
    body = Tex(tex, font_size=font_size)
    fit_to_width(body, width - 0.8)
    content = VGroup(label_mob, body).arrange(DOWN, buff=0.14)
    box = SurroundingRectangle(content, buff=0.24)
    box.set_stroke(RESULT_COLOR, width=2)
    box.set_fill(RESULT_COLOR, opacity=0.08)
    return VGroup(box, content)


def show_intro_cards(
    scene,
    title,
    exercise,
    objective,
    exercise_is_text=False,
    exercise_label=None,
):
    title_mob = scene_title(title)
    exercise_card = (
        exercise_text_block(exercise, label=exercise_label or "Problema")
        if exercise_is_text
        else exercise_block(exercise, label=exercise_label or "Ejercicio")
    )
    objective_card = objective_block(objective)

    cards_group = VGroup(exercise_card, objective_card)
    cards_group.arrange(RIGHT, aligned_edge=UP, buff=0.36)
    cards_group.next_to(title_mob, DOWN, buff=0.26)
    fit_to_width(cards_group, 12.9)

    top_group = VGroup(title_mob, cards_group)
    scene.play(Write(title_mob))
    scene.play(FadeIn(exercise_card), FadeIn(objective_card))
    return top_group, exercise_card, objective_card


def show_key_then_clear(scene, top_group, notes, width=7.2):
    if not notes:
        return None

    key_box = text_panel(
        "Paso clave",
        notes[0],
        width=width,
        font_size=20,
        stroke_color=STEP_COLOR,
    )
    key_box.next_to(top_group, DOWN, buff=0.42)
    scene.play(FadeIn(key_box, shift=0.12 * UP))
    scene.wait(0.75)
    scene.play(FadeOut(key_box))
    return key_box


def _is_verification_line(line):
    normalized = line.replace("\\text{", "").replace("}", "")
    return "Verificacion" in normalized or "Verificación" in normalized


def _split_development_and_verification(steps):
    if steps and _is_verification_line(steps[-1]):
        return steps[:-1], steps[-1]
    return steps, None


def safe_equation_group(lines, top_group, font_size=EQUATION_SIZE, max_width=11.3, buff=0.34):
    equations_group = equation_column(
        lines,
        font_size=font_size,
        max_width=max_width,
        buff=buff,
    )
    equations_group.next_to(top_group, DOWN, buff=0.55)
    equations_group.to_edge(LEFT, buff=0.85)
    return equations_group


def build_equation_steps(lines, top_group, font_size=EQUATION_SIZE, max_width=11.3):
    return safe_equation_group(
        lines,
        top_group,
        font_size=font_size,
        max_width=max_width,
    )


def show_result_box(scene, result, font_size=34):
    final_result = result_box(result, font_size=font_size)
    final_result.to_edge(DOWN, buff=0.38)
    scene.play(FadeIn(final_result, shift=0.15 * UP))
    return final_result


def clear_except(scene, *keepers):
    keeper_ids = {id(mob) for mob in keepers}
    to_clear = [mob for mob in scene.mobjects if id(mob) not in keeper_ids]
    if to_clear:
        scene.play(*[FadeOut(mob) for mob in to_clear])


def run_step_by_step_solution(
    scene,
    title,
    exercise,
    objective,
    steps,
    result,
    notes=None,
    exercise_is_text=False,
    exercise_label=None,
    step_font_size=EQUATION_SIZE,
    result_font_size=34,
):
    """Standard board layout with no explanatory boxes over equations."""
    top_group, _, _ = show_intro_cards(
        scene,
        title,
        exercise,
        objective,
        exercise_is_text=exercise_is_text,
        exercise_label=exercise_label,
    )

    show_key_then_clear(scene, top_group, notes)

    development_steps, verification_step = _split_development_and_verification(steps)
    equations_group = build_equation_steps(
        development_steps,
        top_group,
        font_size=step_font_size,
        max_width=11.3,
    )

    scene.play(Write(equations_group[0]))
    for index in range(1, len(equations_group)):
        scene.play(Write(equations_group[index]))
        scene.wait(0.2)

    verification = None
    if verification_step:
        verification = Tex(verification_step, font_size=max(step_font_size - 4, 24))
        fit_to_width(verification, 11.4)
        verification.next_to(equations_group, DOWN, buff=0.42)
        verification.align_to(equations_group, LEFT)

        if verification.get_bottom()[1] < -FRAME_Y_RADIUS + 1.35:
            scene.play(FadeOut(equations_group))
            verification.next_to(top_group, DOWN, buff=0.75)
            verification.to_edge(LEFT, buff=0.85)

        scene.play(FadeIn(verification, shift=0.12 * UP))
        scene.wait(0.45)
        scene.play(FadeOut(verification))

    scene.play(FadeOut(equations_group))
    show_result_box(scene, result, font_size=result_font_size)
    scene.wait(1.0)
