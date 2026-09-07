# Prompts for the 54-60s clip: a passenger seated in an open-top car.
#
# Separate from make_style_prompts.py because that one describes a car tracked
# laterally at speed. This shot is a portrait -- the subject is a person, the
# vehicle is nearly static in frame, and the camera holds on them. Reusing the
# driving text would describe motion the control does not contain.
#
#   python make_portrait_prompts.py
import json
from pathlib import Path

TRANSFER = Path(__file__).resolve().parent.parent / "cookbooks/cosmos3/generator/transfer"
CONTROL = "../assets/custom/control_seg_54_60.mp4"
FRAMES = 180

# Camera and composition are identical in every variant: they describe the
# framing the control already encodes. Only appearance changes.
CAMERA = {
    "camera_motion": "Slow drift, the camera holding the subject steady in frame with the vehicle nearly stationary relative to it",
    "framing": "Medium close-up",
    "camera_angle": "Eye-level, slightly to the side of the subject",
    "depth_of_field": "Shallow, the subject sharp and the background falling away into soft bokeh",
    "focus": "Locked on the seated passenger",
    "lens_focal_length": "Short telephoto, flattering the face and compressing the background",
}
COMPOSITION = "Medium close-up holding the seated passenger slightly off centre in the upper half of the frame, the car's door line and windscreen frame running across the lower third, background compressed and defocused behind"

BASE = {
    "car": "Deep gloss red lacquer on the door and rear quarter, the paint reading as automotive clear-coat with soft reflections of the sky along its curve, chrome window surround catching a hard highlight, cream leather seating with visible stitching and creasing.",
    "person": "A woman seated in the open car, long hair lifting slightly in the moving air, wearing a fitted dark jacket over a plain top, skin lit warmly from the low sun with soft shadow on the far side of her face, fabric reading as cloth with real weight and fold.",
    "background": "An open road running along a hillside, dry grass and scattered scrub on the slope beyond, a low guard rail catching the light. Distant terrain recedes into warm haze. Everything behind the subject is thrown out of focus into soft rounded bokeh.",
    "lighting": {
        "conditions": "Warm low-angle late-afternoon sunlight, clear sky",
        "direction": "Low and from behind and to the side, rimming the subject's hair and shoulder",
        "shadows": "Soft-edged, with gentle shadow across the far side of the face and a warm bounce filling it from the car's interior",
        "illumination_effect": "Golden backlight producing a bright rim along hair and shoulders, warm flare across the frame edge, and soft specular highlights on paintwork and chrome",
    },
    "colour": "Warm golden light against deep red bodywork and dry gold hillside tones, with cream leather and skin as the mid-values",
    "texture": "Photographic skin detail and individual hair strands catching backlight, fabric weave in the jacket, paint depth and clear-coat reflections, leather grain in the seating",
    "medium": "Live-action video captured on a full-frame digital cinema camera",
    "style": "Photorealistic portrait cinematography",
    "context": "A portrait of a passenger riding in an open-top car on a hillside road at golden hour",
    "action_a": "The passenger sits in the open car, hair lifting in the moving air, the low sun rimming her shoulders as the defocused hillside drifts behind.",
    "action_b": "She turns her head slightly, light shifting across her face, the car's red paintwork catching the sun along its curve.",
    "caption": "At 0:00 a woman sits in an open-top red car on a hillside road, low golden sunlight rimming her hair and shoulders while the background falls away into soft bokeh. By 0:03 her hair lifts in the moving air and warm light shifts across her face. Through to 0:06 the camera holds on her, the red paintwork catching the sun along its curve, the defocused hillside drifting behind.",
    "audio": "A steady engine note carrying from outside the frame, wind moving across the open cabin, and the quiet of an open road. No dialogue or music.",
}

STYLES = {
    "winter": {
        "car": "Deep red lacquer dulled under a film of road salt, meltwater beading along the door line and running back in fine trails, chrome cold and bluish, cream leather seating darkened where snow has melted into it, a dusting of snow caught in the seat stitching.",
        "person": "A woman seated in the open car wearing a cream shearling coat with the collar turned up, a deep red knitted scarf wound at her throat and a soft grey beanie pulled low over her ears, leather gloves resting on the door. Her breath condenses in the freezing air, cheeks and nose flushed with cold, loose hair moving stiffly in the icy draught.",
        "background": "A mountain road under deep snow, drifts banked against a low guard rail and dark conifers rising on the slope beyond, their branches weighted white. Distant peaks fade into flat grey cloud. The background is thrown out of focus into soft cold bokeh.",
        "lighting": {
            "conditions": "Flat overcast winter daylight under heavy snow cloud",
            "direction": "Diffuse and near-directionless, marginally brighter from above",
            "shadows": "Soft, weak and low-contrast, the snow bouncing light back up under the chin and filling the far side of the face",
            "illumination_effect": "Cold blue-grey ambient light at low contrast, snow acting as an enormous reflector, no rim light and no flare",
        },
        "colour": "Desaturated blue-greys and whites of snow and overcast sky, with the red bodywork, red scarf and warm skin tones the only saturated colour in frame",
        "texture": "Condensing breath catching the light, snow crystals in wool fibres, meltwater beading on paintwork, wind-reddened skin, shearling pile and knitted scarf texture",
        "medium": "Live-action video captured on a full-frame digital cinema camera in winter conditions",
        "style": "Photorealistic portrait cinematography, cold and naturalistic",
        "context": "A portrait of a passenger riding in an open-top car on a snowbound mountain road",
        "action_a": "The passenger sits bundled against the cold, breath condensing and streaming away, snow-laden conifers drifting out of focus behind her.",
        "action_b": "She pulls her scarf higher against the wind, flat grey light filling her face, meltwater running back across the red paintwork.",
        "caption": "At 0:00 a woman sits in an open-top red car on a snowbound mountain road, wrapped in a shearling coat and red scarf under flat grey light. By 0:03 her breath condenses and streams away in the freezing air as snow-laden conifers drift out of focus behind. Through to 0:06 the camera holds on her, meltwater beading on the red paintwork, the snow bouncing soft light up into her face.",
        "audio": "A muffled engine note, wind moving cold across the open cabin, and the deadened acoustics of a snow-covered road. No dialogue or music.",
    },
    "night_rain": {
        "car": "Red lacquer darkened and mirror-wet, water sheeting across the door and rear quarter carrying streaked reflections of passing signage, beads racing back along the window line, chrome catching hard coloured highlights, cream leather soaked dark and glistening.",
        "person": "A woman seated in the open car wearing a black satin slip dress under an oversized clear rain poncho beaded with water, gold hoop earrings catching each passing light, hair wet and swept back from her face, water tracking down her bare shoulder. Coloured light shifts across her as signage passes.",
        "background": "A city street at night after heavy rain, shopfront neon and traffic signals reduced to soft coloured orbs in the defocused background, wet asphalt doubling every light source below. Dark building masses rise unlit above the sign line.",
        "lighting": {
            "conditions": "Night after heavy rain, lit entirely by artificial sources",
            "direction": "Multiple hard coloured sources from both sides at low and mid height, shifting as the car moves",
            "shadows": "Hard-edged and multiple, overlapping in different colours across the face and shoulders, deep black where nothing reaches",
            "illumination_effect": "High contrast saturated neon and sodium light raking across wet skin and satin, specular glints in water beads and jewellery, every source doubled in the wet paintwork",
        },
        "colour": "Deep blacks against saturated neon magenta, cyan and amber, the red bodywork reading dark and wine-like under mixed colour temperature, skin caught in shifting coloured light",
        "texture": "Water beading and tracking across skin and satin, wet hair strands separating, sharp specular detail in jewellery and chrome, rain streaks on paintwork, defocused light orbs behind",
        "medium": "Live-action video captured on a full-frame digital cinema camera at high ISO",
        "style": "Photorealistic night portrait cinematography, high contrast and neon-lit",
        "context": "A portrait of a passenger riding in an open-top car through a rain-soaked city at night",
        "action_a": "The passenger sits in the rain-soaked open car, neon light sweeping across her face as signage passes and water tracks down her shoulder.",
        "action_b": "She tilts her head, gold earrings flaring under a passing sign, coloured reflections running along the wet red bodywork.",
        "caption": "At 0:00 a woman sits in an open-top red car on a rain-soaked city street at night, neon reflections streaking across the wet paintwork beside her. By 0:03 coloured light sweeps over her face as signage passes and water tracks down her bare shoulder. Through to 0:06 the camera holds on her, her gold earrings catching each passing source while defocused neon orbs drift behind.",
        "audio": "Rain drumming on bodywork over a low engine note, tyres hissing through standing water, and the wet acoustics of a night street. No dialogue or music.",
    },
    "desert": {
        "car": "Red paintwork bleached and chalky from sun exposure, clear-coat crazed across the horizontal surfaces, a fine pale dust film along the lower door, chrome glaring hot under direct sun, cream leather seating faded and dust-dulled with sun-cracked stitching.",
        "person": "A woman seated in the open car wearing a white linen shirt knotted at the waist, oversized tortoiseshell sunglasses and a wide-brimmed straw hat held down against the wind, a silk scarf tied at her throat streaming behind. Her skin is bright under the vertical sun, hair pulled straight back in the hot dry air.",
        "background": "An open desert highway, cracked hardpan and scattered creosote scrub beyond the road edge, low rock outcrops and distant mesas shimmering in heat haze. The background is defocused into a pale wash of ochre and bleached sky.",
        "lighting": {
            "conditions": "Harsh midday desert sun under a cloudless sky",
            "direction": "Almost directly overhead, marginally forward of vertical",
            "shadows": "Very short and hard-edged, pooled tightly under the hat brim and the chin, nearly black against the bright surroundings",
            "illumination_effect": "Extreme contrast with blown highlights on the hat, shirt and upper bodywork and dense shadow beneath; airborne dust catching light as a pale haze, heat shimmer softening the far background",
        },
        "colour": "Bleached sand, pale ochre and washed-out sky, the red bodywork sun-faded toward orange, white linen and straw reading almost pure white in the glare",
        "texture": "Dust film and paint chalking, sun-cracked leather, linen weave and straw fibre catching hard light, fine dust in the air, heat shimmer distorting distant detail",
        "medium": "Live-action video captured on a full-frame digital cinema camera in desert conditions",
        "style": "Photorealistic portrait cinematography, high-key and sun-bleached",
        "context": "A portrait of a passenger riding in an open-top car on an open desert highway at midday",
        "action_a": "The passenger sits in the open car under vertical sun, holding her hat against the wind as her scarf streams behind and heat haze softens the desert beyond.",
        "action_b": "She turns her head, hard light glaring off the sun-faded paintwork, dust drifting through the air behind her.",
        "caption": "At 0:00 a woman sits in an open-top red car on a desert highway under vertical midday sun, holding a straw hat against the wind. By 0:03 her silk scarf streams back from her throat as heat haze shimmers over the defocused hardpan behind. Through to 0:06 the camera holds on her, hard light glaring off sun-faded paintwork and blowing out the white linen of her shirt.",
        "audio": "A steady engine note carrying in dry open air, wind rushing hot across the open cabin, and the emptiness of a road with no other traffic. No dialogue or music.",
    },
}


def build(style, seconds):
    return {
        "subjects": [
            {
                "description": "A woman seated as a passenger in an open-top convertible car, the roof down.",
                "appearance_details": style["person"],
                "position": "Slightly off centre in the upper half of the frame, seen from the side at eye level",
            },
            {
                "description": "The open-top car she is riding in, seen in part behind and around her.",
                "appearance_details": style["car"],
                "position": "Framing the subject, its door line and windscreen surround crossing the lower third of the image",
            },
        ],
        "background_setting": style["background"],
        "lighting": style["lighting"],
        "aesthetics": {
            "composition": COMPOSITION,
            "color_scheme": style["colour"],
            "visual_texture": style["texture"],
        },
        "cinematography": dict(CAMERA),
        "style_medium": style["medium"],
        "artistic_style": style["style"],
        "context": style["context"],
        "actions": [
            {"time": "0:00-0:03", "description": style["action_a"]},
            {"time": "0:03-0:06", "description": style["action_b"]},
        ],
        "text_and_signage_elements": [],
        "segments": [
            {
                "segment_index": 0,
                "time_range": f"0:00-0:0{seconds}",
                "description": style["context"],
                "key_changes": "Light shifts across the subject's face and the bodywork as the car moves; the defocused background drifts behind without ever resolving into detail.",
            }
        ],
        "transitions": [],
        "temporal_caption": style["caption"],
        "audio_description": style["audio"],
        "resolution": {"W": 1280, "H": 720},
        "aspect_ratio": "16,9",
        "duration": f"{seconds}s",
        "fps": 30,
    }


def main():
    reference = json.loads((TRANSFER / "assets/seg/prompt.json").read_text(encoding="utf-8"))
    spec_base = json.loads((TRANSFER / "specs/seg.json").read_text(encoding="utf-8"))
    seconds = FRAMES // 30

    for name, style in [("base", BASE), *STYLES.items()]:
        prompt = build(style, seconds)
        if set(prompt) != set(reference):
            raise SystemExit(f"{name}: schema does not match the shipped prompt")
        suffix = "" if name == "base" else f"_{name}"
        prompt_path = TRANSFER / f"assets/custom/prompt_portrait{suffix}.json"
        prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

        spec = json.loads(json.dumps(spec_base))
        spec["name"] = f"transfer_seg_portrait{suffix}"
        spec["num_frames"] = FRAMES
        # One pass rather than chained chunks: 180 frames is within the range
        # the framework calls acceptable, so there is no need to hand off.
        spec["num_video_frames_per_chunk"] = FRAMES
        spec["prompt_path"] = f"../assets/custom/prompt_portrait{suffix}.json"
        spec["negative_prompt_file"] = "../assets/custom/negative_prompt.json"
        spec["seg"] = {"control_path": CONTROL}
        spec_path = TRANSFER / f"specs/seg_portrait{suffix}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"  {name:11s} {prompt_path.name}  +  {spec_path.name}")


if __name__ == "__main__":
    main()
