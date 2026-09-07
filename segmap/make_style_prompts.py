# Generate style variants of the driving-clip prompt, one file per style.
#
# The control video is unchanged across all of them: it carries the geometry and
# motion. Only appearance varies, which is what transfer is for. Camera and
# composition fields are therefore identical in every variant -- they describe
# the motion the control already encodes, so varying them would fight it.
import json
from pathlib import Path

TRANSFER = Path(__file__).resolve().parent.parent / "cookbooks/cosmos3/generator/transfer"
BASE = TRANSFER / "assets/custom/prompt_drive_a.json"

STYLES = {
    "winter": {
        "car": "Deep gloss yellow lacquer dulled by a film of road salt, grey slush thrown up behind the wheel arches and crusted along the sills, a wet sheen across the bonnet where snowmelt runs off, chrome bumpers cold and bluish with meltwater beading on them, an illuminated roof sign glowing warm against the grey air, winter tyres cutting dark tracks through surface snow.",
        "people": "The woman in the rear seat wears a cream shearling coat over a chunky roll-neck sweater, a deep red knitted scarf streaming backward and a matching beanie pulled low, leather gloves on the door frame. Her companion is in a quilted parka with the hood down. Breath condenses in the freezing air and streams away behind them, cheeks reddened by cold.",
        "background": "A wide boulevard under deep snow, banks of ploughed snow heaped along the kerb and grey slush tracked across the carriageway in dark wheel lines. Bare deciduous trees stand with snow lying along their branches. A low wall runs behind the pavement, its coping capped white. Buildings recede into cold grey haze, roofs snow-covered, lights burning in a few windows against the dim afternoon.",
        "lighting": {
            "conditions": "Flat overcast winter daylight under heavy snow cloud",
            "direction": "Diffuse and near-directionless, marginally brighter from above",
            "shadows": "Soft, weak and short, with only a faint contact shadow beneath the car in the compacted snow",
            "illumination_effect": "Cold blue-grey ambient light at low contrast, the snow acting as a huge reflector filling the underside of the car; the roof sign and tail lights are the only warm sources",
        },
        "colour": "Desaturated blue-greys and whites of snow and overcast sky, dark wet asphalt showing through in wheel tracks, with the yellow bodywork and warm tail lights the only saturated colour in frame",
        "texture": "Snow crystal detail on the banks, slush spray caught mid-air, salt film and water beading on paintwork, wet tyre tread throwing fine mist",
        "medium": "Live-action video captured on a full-frame digital cinema camera from a tracking vehicle in winter conditions",
        "style": "Photorealistic automotive cinematography, cold and naturalistic",
        "context": "A tracking shot of a convertible taxi driving through a snow-covered boulevard on an overcast winter afternoon",
        "action_a": "The taxi travels left to right through packed snow, the camera holding pace alongside as slush sprays from the tyres and snow-laden trees pass behind.",
        "action_b": "The car continues across the frame, the passengers' breath streaming backward in the cold air, meltwater running back across the bonnet.",
        "caption": "At 0:00 a yellow taxi moves left to right along a snow-covered boulevard under flat grey light, the camera tracking alongside as slush sprays from beneath the tyres. By 0:01 the wheels cut dark lines through surface snow and the woman's red scarf streams backward in the freezing air. Through to 0:02 the car holds its place in frame, salt film and meltwater glistening on the yellow paintwork, while ploughed snow banks and bare trees sweep past behind.",
        "audio": "A muffled engine note under load, tyres compressing packed snow with a crunching hiss, wind across the open cabin, and the dulled acoustics of a snow-covered street. No dialogue or music.",
    },
    "night_rain": {
        "car": "Yellow lacquer darkened and mirror-wet, the body sheeted in running water carrying streaked reflections of shopfront neon, beads racing backward across the bonnet, chrome catching hard coloured highlights, an illuminated roof sign flaring against the dark, tyres throwing fine spray from standing water.",
        "people": "The woman in the rear seat wears a black satin slip dress under an oversized transparent rain poncho beaded with water, gold hoop earrings catching passing signage, hair wet and swept back. Her companion is in a soaked dark denim jacket. Coloured light shifts across them as each sign passes.",
        "background": "A city boulevard at night after heavy rain, the asphalt a black mirror carrying long vertical smears of red, green and cyan from shopfront signage and traffic lights. Standing water pools along the gutter and wet pavement reflects the storefronts. A low wall runs behind, its face darkened by rain. Buildings rise into unlit darkness above the sign line.",
        "lighting": {
            "conditions": "Night after heavy rain, lit entirely by artificial sources",
            "direction": "Multiple hard sources at low and mid height from both sides, no sky light",
            "shadows": "Multiple overlapping coloured shadows, hard-edged and shifting as the car passes each source, with deep black shadow beneath the vehicle",
            "illumination_effect": "High contrast saturated light from neon and sodium sources raking across wet paintwork, specular streaks running the length of the body, every source doubled in the mirrored road surface",
        },
        "colour": "Deep blacks and wet asphalt against saturated neon magenta, cyan, red and amber, the yellow bodywork reading warm and desaturated under mixed colour temperature",
        "texture": "Rain streaks running across paintwork, water beading and shearing off in the airflow, sharp specular detail in wet chrome, fine spray suspended behind the wheels, mirror reflections broken by ripples",
        "medium": "Live-action video captured on a full-frame digital cinema camera at high ISO from a tracking vehicle",
        "style": "Photorealistic night automotive cinematography, high contrast and neon-lit",
        "context": "A tracking shot of a convertible taxi driving through a rain-soaked city boulevard at night",
        "action_a": "The taxi travels left to right across the wet road, the camera holding pace as neon reflections sweep along the flank and spray lifts from the tyres.",
        "action_b": "The car continues through standing water, coloured light raking across the bodywork and doubling in the mirrored asphalt below.",
        "caption": "At 0:00 a yellow taxi moves left to right along a rain-soaked boulevard at night, the camera tracking alongside as neon reflections streak across its wet flank. By 0:01 it passes through standing water, spray lifting behind the wheels while the woman's gold earrings catch a passing sign. Through to 0:02 the car holds frame as shopfront signage sweeps past behind, each source smeared long into the black road surface.",
        "audio": "Engine note over the hiss of tyres displacing standing water, rain drumming on bodywork, wind across the open cabin, and the wet acoustics of a night street. No dialogue or music.",
    },
    "desert": {
        "car": "Yellow paintwork bleached and chalky from sun exposure, a fine pale dust film across the lower panels and rear, clear-coat crazed on the horizontal surfaces, chrome hot and glaring under direct sun, an illuminated roof sign washed out to near-invisibility, tyres coated in pale dust that lifts in a plume behind.",
        "people": "The woman in the rear seat wears a white linen shirt knotted at the waist over high-waisted denim, oversized tortoiseshell sunglasses and a wide-brimmed straw hat held against the wind, a silk scarf tied at her throat streaming behind. Her companion is in a faded short-sleeved shirt. Both squint against the vertical glare.",
        "background": "An open desert highway running dead straight, flanked by cracked hardpan, scattered creosote scrub and low rock outcrops. The asphalt is sun-bleached grey with faded centre markings and long tar seams. A dust plume trails behind the vehicle. Distant mesas shimmer in heat haze under a hard, almost white sky.",
        "lighting": {
            "conditions": "Harsh midday desert sun under a cloudless sky",
            "direction": "Almost directly overhead, marginally forward of vertical",
            "shadows": "Very short, hard-edged and nearly black, pooled tightly beneath the car and along the underside of the body",
            "illumination_effect": "Extreme contrast with blown highlights on upper surfaces and dense shadow beneath; heat shimmer distorting the road at distance, airborne dust catching light as a pale haze",
        },
        "colour": "Bleached sand, pale ochre and washed grey asphalt under an almost white sky, the yellow bodywork tending toward the desert palette rather than contrasting with it",
        "texture": "Dust film and paint chalking on the bodywork, cracked hardpan and asphalt aggregate, airborne dust catching light, heat shimmer distorting distant detail",
        "medium": "Live-action video captured on a full-frame digital cinema camera from a tracking vehicle in desert conditions",
        "style": "Photorealistic automotive cinematography, high-key and sun-bleached",
        "context": "A tracking shot of a convertible taxi driving along an open desert highway at midday",
        "action_a": "The taxi travels left to right along the empty highway, the camera holding pace as a dust plume lifts behind and heat shimmer distorts the road ahead.",
        "action_b": "The car continues across the frame, hard overhead sun glaring off the chrome, scrub and rock passing behind in the dry air.",
        "caption": "At 0:00 a dust-covered yellow taxi moves left to right along an empty desert highway under vertical midday sun, the camera tracking alongside. By 0:01 a pale dust plume lifts behind the rear wheels and the woman's silk scarf streams back from her throat. Through to 0:02 the car holds frame, hard light glaring off chrome and bleached paintwork, while scrub and distant mesas sweep past in the dry haze.",
        "audio": "A steady engine note carrying in dry open air, tyres on hot coarse asphalt, wind rushing across the open cabin, and the emptiness of a road with no other traffic. No dialogue or music.",
    },
}


def main():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    spec_base = json.loads((TRANSFER / "specs/seg_drive_a.json").read_text(encoding="utf-8"))

    for name, style in STYLES.items():
        prompt = json.loads(json.dumps(base))
        prompt["subjects"][0]["appearance_details"] = style["car"]
        prompt["subjects"][1]["appearance_details"] = style["people"]
        prompt["background_setting"] = style["background"]
        prompt["lighting"] = style["lighting"]
        prompt["aesthetics"]["color_scheme"] = style["colour"]
        prompt["aesthetics"]["visual_texture"] = style["texture"]
        prompt["style_medium"] = style["medium"]
        prompt["artistic_style"] = style["style"]
        prompt["context"] = style["context"]
        prompt["actions"][0]["description"] = style["action_a"]
        prompt["actions"][1]["description"] = style["action_b"]
        prompt["segments"][0]["description"] = style["context"]
        prompt["temporal_caption"] = style["caption"]
        prompt["audio_description"] = style["audio"]
        if set(prompt) != set(base):
            raise SystemExit(f"{name}: schema drifted from the base prompt")

        prompt_path = TRANSFER / f"assets/custom/prompt_drive_a_{name}.json"
        prompt_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

        spec = json.loads(json.dumps(spec_base))
        spec["name"] = f"transfer_seg_drive_a_{name}"
        spec["prompt_path"] = f"../assets/custom/prompt_drive_a_{name}.json"
        spec_path = TRANSFER / f"specs/seg_drive_a_{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"  {name:11s} {prompt_path.name}  +  {spec_path.name}")


if __name__ == "__main__":
    main()
