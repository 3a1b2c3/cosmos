# Prompts for the taxi gameplay capture: a chase camera behind the vehicle.
#
# Separate from the other generators because the camera is different. The drive
# prompts describe a car tracked laterally in profile; the portrait prompts
# describe a person held in frame. Here the camera sits behind and above the
# taxi looking down the street, so composition, framing and what sweeps past are
# all different, and reusing either would describe motion the control does not
# contain.
#
#   python make_taxi_prompts.py --control ../../data/taxi/control_depth_clean.mp4 --hint depth --frames 200
import argparse
import json
from pathlib import Path

TRANSFER = Path(__file__).resolve().parent.parent / "cookbooks/cosmos3/generator/transfer"

# Identical across every variant: they describe the camera the control already
# encodes. Only appearance changes between styles.
CAMERA = {
    "camera_motion": "Chase camera following directly behind the vehicle at matching speed, holding it centred while the street flows past on both sides",
    "framing": "Medium-wide shot from behind",
    "camera_angle": "Elevated, a little above roof height, looking down the street over the car",
    "depth_of_field": "Deep, the vehicle and near street sharp with distance softening into atmosphere",
    "focus": "Locked on the car and the road immediately ahead of it",
    "lens_focal_length": "Wide-angle, opening up the street ahead",
}
COMPOSITION = "Chase framing with the car centred in the lower half of the image, its rear deck and tail filling the foreground, the street receding to a vanishing point ahead and building frontages closing in from both edges"

BASE = {
    "car": "A long yellow convertible taxi seen from behind, roof down, deep gloss lacquer over steel with a black-and-white chequer band along the flanks, chrome bumper and tail trim pitted with age, red upholstery visible in the open cabin, tail lights and licence plate catching the light, road film along the lower panels.",
    "people": "A passenger in the open rear seat, hair and clothing pulled backward by the airflow, lit by the same daylight as the street.",
    "background": "A narrow cobbled street through an old European town, timber-framed houses with white render and dark beams rising on both sides, shopfronts with hanging signs and awnings at street level, street trees and lamp posts along the kerb. Pedestrians move along the pavements. The cobbles are uneven and worn, with tram lines and drain covers set into them.",
    "lighting": {
        "conditions": "Bright daylight under a lightly clouded sky",
        "direction": "High and from the front-left, throwing the shadow of the car back toward the camera",
        "shadows": "Crisp-edged shadows from buildings falling across the street, with a firm contact shadow beneath the car anchoring it to the cobbles",
        "illumination_effect": "Clean directional light with specular highlights along the chrome and the car's shoulder line, façades brightly lit on one side of the street and in shade on the other",
    },
    "colour": "Saturated yellow bodywork against warm render, dark timber framing and grey cobbles, with red upholttery and shop signage as accents",
    "texture": "Individual cobblestones with worn edges and mortar joints, render and timber grain on the façades, paint depth and chrome pitting on the car, fabric and hair movement in the airflow",
    "medium": "Live-action video captured on a full-frame digital cinema camera from a following vehicle",
    "style": "Photorealistic automotive cinematography",
    "context": "A chase shot following a convertible taxi through an old European town",
    "action_a": "The taxi accelerates down the cobbled street with the camera holding station behind it, façades and pedestrians sweeping past on both sides.",
    "action_b": "The car continues ahead, its body settling over the uneven cobbles, shadow tracking beneath it as the street opens toward a junction.",
    "caption": "At 0:00 the camera follows a yellow convertible taxi down a narrow cobbled street between timber-framed houses, holding station directly behind it. By 0:03 façades, shopfronts and pedestrians sweep past on both sides as the car settles over the uneven surface. Through to 0:06 the street opens ahead, the car's shadow tracking beneath it and chrome catching the daylight.",
    "audio": "Engine note under acceleration, tyres drumming over cobblestones, the close acoustics of a narrow street, and faint pedestrian noise. No dialogue or music.",
}

STYLES = {
    "winter": {
        "car": "A long yellow convertible taxi seen from behind, its lacquer dulled under road salt, grey slush crusted along the sills and thrown up behind the rear wheels, chrome cold and bluish with meltwater beading, red upholstery dusted with snow, tail lights burning warm against the grey air.",
        "people": "A passenger bundled in a heavy coat and knitted scarf, breath condensing and streaming backward in the freezing air.",
        "background": "A narrow cobbled street through an old European town under snow, timber-framed houses with snow lying along every ledge and window sill, banks of ploughed snow heaped at the kerb and dark slush tracked down the middle of the carriageway. Bare street trees stand black against the white. Shopfront lights burn warm in the dim afternoon.",
        "lighting": {
            "conditions": "Flat overcast winter daylight under heavy snow cloud",
            "direction": "Diffuse and near-directionless, marginally brighter from above",
            "shadows": "Weak and soft, with only a faint contact shadow beneath the car in the compacted snow",
            "illumination_effect": "Cold blue-grey ambient light at low contrast, snow acting as a huge reflector filling the underside of the car; shopfront windows and tail lights are the only warm sources",
        },
        "colour": "Desaturated blue-greys and whites of snow and overcast sky, dark wet cobbles showing through in wheel tracks, with the yellow bodywork and warm shop lights the only saturated colour",
        "texture": "Snow crystals along ledges and roof tiles, slush spray caught behind the wheels, salt film and meltwater on paintwork, breath condensing in the cold",
        "medium": "Live-action video captured on a full-frame digital cinema camera from a following vehicle in winter conditions",
        "style": "Photorealistic automotive cinematography, cold and naturalistic",
        "context": "A chase shot following a convertible taxi through a snowbound old European town",
        "action_a": "The taxi moves down the snow-covered street with the camera behind it, slush spraying from the rear wheels as snow-laden façades pass on both sides.",
        "action_b": "The car continues ahead through compacted snow, its wheels cutting dark tracks, the passenger's breath streaming backward in the freezing air.",
        "caption": "At 0:00 the camera follows a yellow taxi down a snow-covered cobbled street between timber-framed houses under flat grey light. By 0:03 slush sprays from the rear wheels and the passenger's breath condenses in the freezing air. Through to 0:06 the wheels cut dark tracks through packed snow as warm shopfront windows pass on both sides.",
        "audio": "A muffled engine note, tyres compressing packed snow with a crunching hiss, and the deadened acoustics of a snow-covered street. No dialogue or music.",
    },
    "night_rain": {
        "car": "A long yellow convertible taxi seen from behind, lacquer darkened and mirror-wet, water sheeting off the rear deck and beading across the boot lid, chrome catching hard coloured highlights, red upholstery soaked dark, tail lights flaring and doubling in the wet surface below.",
        "people": "A passenger in a soaked dark jacket, hair flattened by rain, lit intermittently in shifting colour as signage passes.",
        "background": "A narrow cobbled street through an old European town at night after heavy rain, the cobbles a black mirror carrying long smears of light from shopfront signage and street lamps. Timber-framed façades rise into darkness above the lit ground floor. Standing water pools between the stones and along the gutter.",
        "lighting": {
            "conditions": "Night after heavy rain, lit by shop windows and street lamps",
            "direction": "Multiple sources at low and mid height from both sides, no sky light",
            "shadows": "Multiple overlapping shadows, hard-edged and shifting as the car passes each source, deep black beneath the vehicle",
            "illumination_effect": "High contrast warm sodium and shop-window light raking across wet paintwork, every source doubled in the mirrored cobbles, tail lights bleeding red across the wet surface behind",
        },
        "colour": "Deep blacks and wet stone against warm amber street lighting and coloured shop signage, the yellow bodywork reading dark and rich under mixed colour temperature",
        "texture": "Rain streaking across paintwork, water beading and shearing off in the airflow, sharp specular detail in wet chrome, mirrored reflections in cobbles broken by ripples",
        "medium": "Live-action video captured on a full-frame digital cinema camera at high ISO from a following vehicle",
        "style": "Photorealistic night automotive cinematography, high contrast and lamp-lit",
        "context": "A chase shot following a convertible taxi through a rain-soaked old European town at night",
        "action_a": "The taxi moves down the wet cobbled street with the camera behind it, spray lifting from the rear wheels and lamp light streaking along its flanks.",
        "action_b": "The car continues ahead through standing water, its tail lights bleeding red across the mirrored stones behind it.",
        "caption": "At 0:00 the camera follows a yellow taxi down a rain-soaked cobbled street at night, shopfront light streaking across its wet rear deck. By 0:03 spray lifts from the rear wheels and every lamp doubles in the mirrored stones. Through to 0:06 the car's tail lights bleed red across the wet surface as darkened timber façades pass on both sides.",
        "audio": "Engine note over the hiss of tyres on wet cobbles, rain drumming on bodywork, and the close wet acoustics of a narrow street. No dialogue or music.",
    },
    "autumn": {
        "car": "A long yellow convertible taxi seen from behind, lacquer warm and slightly dulled, a scatter of fallen leaves caught in the rear deck and against the base of the windscreen, chrome catching low amber light, red upholstery deepened by the warm cast, damp road film along the lower panels.",
        "people": "A passenger in a wool coat with the collar up, hair lifting in the cool air, lit warmly from the low sun ahead.",
        "background": "A narrow cobbled street through an old European town in autumn, street trees in deep gold and rust dropping leaves that drift and skitter across the stones. Timber-framed houses stand warm in low sun on one side and in blue shade on the other. Damp patches darken the cobbles, and drifts of leaves gather along the kerb.",
        "lighting": {
            "conditions": "Low autumn sun under a clear sky, late in the afternoon",
            "direction": "Very low and from directly ahead, backlighting the street toward the camera",
            "shadows": "Long shadows stretching back toward the camera from every object, with the car's own shadow running ahead of it",
            "illumination_effect": "Strong warm backlight rimming the car and the falling leaves, haze and flare where the sun sits low in frame, deep cool shadow filling the shaded side of the street",
        },
        "colour": "Deep golds, rust and amber of autumn foliage against warm render and grey cobbles, the yellow bodywork blending into a broadly warm palette with cool blue shade as counterpoint",
        "texture": "Individual leaves tumbling and catching backlight, damp cobbles with a low sheen, warm grain in render and timber, dust and pollen suspended in the low sun",
        "medium": "Live-action video captured on a full-frame digital cinema camera from a following vehicle",
        "style": "Photorealistic automotive cinematography, warm and backlit",
        "context": "A chase shot following a convertible taxi through an old European town on a low-sun autumn afternoon",
        "action_a": "The taxi moves down the cobbled street with the camera behind it, fallen leaves lifting and tumbling in its wake as low sun flares between the buildings.",
        "action_b": "The car continues ahead into the backlight, its long shadow running toward the camera as gold foliage passes on both sides.",
        "caption": "At 0:00 the camera follows a yellow taxi down a cobbled street lined with gold autumn trees, low sun flaring between the timber-framed houses ahead. By 0:03 fallen leaves lift and tumble in the car's wake across the damp stones. Through to 0:06 the car drives into the backlight, its long shadow stretching back toward the camera.",
        "audio": "Engine note under acceleration, tyres drumming over cobblestones, dry leaves skittering in the wake, and the close acoustics of a narrow street. No dialogue or music.",
    },
}


def build(style, seconds):
    return {
        "subjects": [
            {
                "description": "A yellow convertible taxi cab with its roof down, driving away from the camera down a narrow street.",
                "appearance_details": style["car"],
                "position": "Centred in the lower half of the frame, seen from directly behind, receding down the street",
            },
            {
                "description": "A passenger riding in the open rear of the car.",
                "appearance_details": style["people"],
                "position": "In the open cabin at the centre of the vehicle, upper middle of the car's silhouette",
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
            {"time": f"0:00-0:0{seconds // 2}", "description": style["action_a"]},
            {"time": f"0:0{seconds // 2}-0:0{seconds}", "description": style["action_b"]},
        ],
        "text_and_signage_elements": [],
        "segments": [
            {
                "segment_index": 0,
                "time_range": f"0:00-0:0{seconds}",
                "description": style["context"],
                "key_changes": "Light travels along the car's bodywork as it moves; façades, street furniture and pedestrians sweep past on both sides while the vehicle holds the centre of frame.",
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", required=True, help="path the spec should reference")
    parser.add_argument("--hint", default="depth", choices=("depth", "seg", "edge"),
                        help="which control key the spec uses")
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    reference = json.loads((TRANSFER / "assets/seg/prompt.json").read_text(encoding="utf-8"))
    spec_base = json.loads((TRANSFER / "specs/seg.json").read_text(encoding="utf-8"))
    seconds = max(1, round(args.frames / 30))

    for name, style in [("base", BASE), *STYLES.items()]:
        prompt = build(style, seconds)
        if set(prompt) != set(reference):
            raise SystemExit(f"{name}: schema does not match the shipped prompt")
        suffix = "" if name == "base" else f"_{name}"
        prompt_path = TRANSFER / f"assets/custom/prompt_taxi{suffix}.json"
        prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

        spec = json.loads(json.dumps(spec_base))
        spec["name"] = f"transfer_taxi_{args.hint}{suffix}"
        spec["num_frames"] = args.frames
        # One pass rather than chained chunks, provided frames stays within the
        # range the framework calls acceptable.
        spec["num_video_frames_per_chunk"] = args.frames
        spec["prompt_path"] = f"../assets/custom/prompt_taxi{suffix}.json"
        spec["negative_prompt_file"] = "../assets/custom/negative_prompt.json"
        spec.pop("seg", None)
        spec[args.hint] = {"control_path": args.control}
        spec_path = TRANSFER / f"specs/taxi_{args.hint}{suffix}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"  {name:11s} {prompt_path.name}  +  {spec_path.name}")


if __name__ == "__main__":
    main()
