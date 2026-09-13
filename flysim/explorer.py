"""Interactive brain explorer: stimulate a sense, get spikes of the whole brain frame by frame."""
import base64
import threading

import numpy as np
import torch

from . import body, connectome, neurons
from .brain import FlyBrain
from .config import LIFParams

BIN_MS = 5.0

# Outputs we can name, with what they do in a real fly.
KNOWN_OUTPUTS = {
    "MN9": "вытягивает хоботок — «хочу есть»",
    "DNp01": "гигантское волокно — прыжок и взлёт, побег",
    "DNb05": "поворот (в модели — на сторону зрительного стимула)",
    "DNa01": "поворот при ходьбе",
    "DNa02": "поворот при ходьбе",
    "MDN": "«лунная походка» — идёт назад",
    "DNp09": "ходьба вперёд",
}
MOTOR_SUBCLASS = {
    "proboscis_motor_neuron": "мышцы хоботка",
    "ingestion_motor_neuron": "глотание",
    "neck_motor_neuron": "поворот головы",
    "antennal_motor_neuron": "движение усиков",
    "eye_motor_neuron": "мышцы сетчатки",
    "haustellum_motor_neuron": "кончик хоботка",
    "salivary_motor_neuron": "слюна",
    "crop_motor_neuron": "зоб",
}


class Explorer:
    def __init__(self):
        self.con = connectome.load()
        ann = connectome.annotations()
        ann = ann[~ann.index.duplicated()]
        self.ann = ann.reindex(self.con.ids)
        self.lock = threading.Lock()
        self.brain = None
        self._build_layout()
        self._build_presets()

    # --- static data -------------------------------------------------------
    def _build_layout(self):
        a = self.ann
        x, y = a.pos_x.to_numpy(float), a.pos_y.to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        x0, x1, y0, y1 = x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max()
        span = max(x1 - x0, y1 - y0)
        self.x = np.where(ok, (x - x0) / span, -1).astype(np.float32)
        self.y = np.where(ok, (y - y0) / span, -1).astype(np.float32)
        self.aspect = float((y1 - y0) / (x1 - x0))
        classes = a.super_class.fillna("unknown")
        self.class_names = sorted(classes.unique())
        self.class_code = classes.map({c: i for i, c in enumerate(self.class_names)}).to_numpy(np.uint8)
        types = a.cell_type.fillna(a.cell_class).fillna(classes).astype(str)
        self.type_names = sorted(types.unique())
        self.type_code = types.map({t: i for i, t in enumerate(self.type_names)}).to_numpy(np.uint16)
        self.types = types.to_numpy()

    def layout_bytes(self) -> bytes:
        n = self.con.n
        pad = (-n) % 4
        return (self.x.tobytes() + self.y.tobytes() + self.class_code.tobytes() + b"\0" * pad
                + self.type_code.tobytes())

    def _build_presets(self):
        a, con = self.ann, self.con
        side = a.side
        dn_idx = np.flatnonzero((a.super_class == "descending").to_numpy())
        reach = body.dn_reach(con, dn_idx)

        def where(mask):
            return np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()

        def ranked(mask, k):
            idx = np.flatnonzero(np.asarray(mask, dtype=bool))
            return idx[np.argsort(-reach[idx], kind="stable")][:k].tolist()

        vpn = a.super_class == "visual_projection"
        gust = a.cell_class == "gustatory"
        self.presets = {
            "sugar": ("🍬 Сахар на хоботок", "вкусовые нейроны сахара",
                      [con.index_of[i] for i in neurons.SUGAR_GRN if i in con.index_of]),
            "bitter": ("🍋 Горькое", "вкусовые нейроны горького", where(gust & (a.cell_sub_class == "bitter"))),
            "salt": ("🧂 Немного соли", "вкусовые нейроны низкой солёности", where(gust & (a.cell_sub_class == "low-salt"))),
            "vision_left": ("👁️ Что-то слева", "зрительные проекционные нейроны левого глаза", ranked(vpn & (side == "left"), 60)),
            "vision_right": ("👁️ Что-то справа", "зрительные проекционные нейроны правого глаза", ranked(vpn & (side == "right"), 60)),
            "looming": ("🦅 Надвигается тень", "детекторы приближения LC4 и LPLC2", where(a.cell_type.isin(["LC4", "LPLC2"]))),
            "wind": ("💨 Ветер", "нейроны Джонстонова органа (ветер, гравитация)", where(a.cell_sub_class == "wind_gravity")),
            "sound": ("🎵 Звук", "слуховые нейроны Джонстонова органа", where(a.cell_sub_class == "auditory")),
            "touch_head": ("👆 Касание головы", "механорецепторы щетинок головы", where(a.cell_sub_class == "head bristle")),
            "smell": ("👃 Запах (эпилепсия!)", "обонятельные рецепторы — в этой модели запускают разгон", ranked((a.cell_class == "olfactory") & (side == "left"), 60)),
        }

    def meta(self) -> dict:
        return {
            "n": self.con.n, "aspect": self.aspect, "bin_ms": BIN_MS,
            "classes": self.class_names, "types": self.type_names,
            "presets": [{"id": k, "name": v[0], "desc": v[1], "count": len(v[2])} for k, v in self.presets.items()],
        }

    # --- simulation --------------------------------------------------------
    @torch.no_grad()
    def stimulate(self, stims: list[dict], ms: float = 300.0, stim_ms: float = 150.0) -> dict:
        """stims: [{"preset": id, "hz": rate}]. Stimulus is on for the first stim_ms."""
        ms = float(min(max(ms, 50), 1000))
        groups = [(self.presets[s["preset"]][2], float(s.get("hz", 150))) for s in stims if s.get("preset") in self.presets]
        with self.lock:
            if self.brain is None:
                self.brain = FlyBrain(self.con, batch=1, params=LIFParams(dt=0.5))
            brain = self.brain
            brain.resize(1)
            dev = brain.device
            idx = sorted({i for g, _ in groups for i in g})
            rates = torch.zeros(len(idx), 1, device=dev)
            row = {n: r for r, n in enumerate(idx)}
            for g, hz in groups:
                rates[[row[i] for i in g], 0] = hz
            idx_t = torch.tensor(idx, device=dev, dtype=torch.long) if idx else None

            steps, per_bin = int(ms / brain.p.dt), int(BIN_MS / brain.p.dt)
            stim_steps = int(stim_ms / brain.p.dt)
            counts = torch.zeros(brain.n, device=dev)
            bin_spikes = torch.zeros(brain.n, dtype=torch.bool, device=dev)
            frames, total = [], np.zeros(brain.n, dtype=np.int32)
            for step in range(steps):
                on = idx_t is not None and step < stim_steps
                s = brain.step(idx_t, rates) if on else brain.step()
                s = s[:, 0]
                bin_spikes |= s
                counts += s
                if (step + 1) % per_bin == 0:
                    frames.append(torch.nonzero(bin_spikes).squeeze(1).to(torch.int32).cpu().numpy())
                    bin_spikes.zero_()
            total = counts.cpu().numpy()

        sizes = np.array([len(f) for f in frames], dtype=np.int32)
        flat = np.concatenate(frames) if frames else np.zeros(0, np.int32)
        by_class = [[int(np.isin(self.class_code[f], [c]).sum()) for c in range(len(self.class_names))] for f in frames]
        return {
            "ms": ms, "stim_ms": stim_ms, "bin_ms": BIN_MS,
            "frame_sizes": base64.b64encode(sizes.tobytes()).decode(),
            "spikes": base64.b64encode(flat.astype(np.int32).tobytes()).decode(),
            "by_class": by_class,
            "active": int((total > 0).sum()),
            "outputs": self._outputs(total, ms),
            "stimulated": len(idx),
        }

    def _outputs(self, total: np.ndarray, ms: float) -> list:
        a, out = self.ann, []
        rate = total / (ms / 1000.0)
        for t, what in KNOWN_OUTPUTS.items():
            if t == "MN9":
                sel = [self.con.index_of[neurons.MN9]]
            else:
                sel = np.flatnonzero((a.cell_type == t).to_numpy())
            for i in sel:
                if total[i] > 0:
                    out.append({"name": f"{t} {self._side(i)}", "hz": round(float(rate[i]), 1), "what": what, "index": int(i)})
        motor = np.flatnonzero((a.super_class == "motor").to_numpy())
        for i in motor[np.argsort(-total[motor])][:12]:
            if total[i] > 0 and i != self.con.index_of[neurons.MN9]:
                sub = a.cell_sub_class.iloc[i]
                out.append({"name": f"{self.types[i]} {self._side(i)}", "hz": round(float(rate[i]), 1),
                            "what": MOTOR_SUBCLASS.get(sub, "моторный нейрон"), "index": int(i)})
        dn = np.flatnonzero((a.super_class == "descending").to_numpy())
        known = {o["index"] for o in out}
        for i in dn[np.argsort(-total[dn])][:8]:
            if total[i] > 0 and i not in known:
                out.append({"name": f"{self.types[i]} {self._side(i)}", "hz": round(float(rate[i]), 1),
                            "what": "нисходящий нейрон → в брюшную нервную цепочку", "index": int(i)})
        return sorted(out, key=lambda o: -o["hz"])

    def _side(self, i: int) -> str:
        s = self.ann.side.iloc[i]
        return {"left": "Л", "right": "П"}.get(s, "")
