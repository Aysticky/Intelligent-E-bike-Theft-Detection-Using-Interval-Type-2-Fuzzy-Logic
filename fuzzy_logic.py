import numpy as np

# Membership Functions
def tri_mf(a, b, c, height=1.0):
    def mf(x):
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return height
        elif x < b:
            return height * (x - a) / (b - a) if b != a else 0.0
        else:
            return height * (c - x) / (c - b) if c != b else 0.0
    return mf

def trap_mf(a, b, c, d, height=1.0):
    def mf(x):
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return height
        elif a < x < b:
            return height * (x - a) / (b - a) if b != a else 0.0
        else:
            return height * (d - x) / (d - c) if d != c else 0.0
    return mf

# IT2FS Class
class IT2FS:
    def __init__(self, lower_mf, upper_mf):
        self.lower_mf = lower_mf
        self.upper_mf = upper_mf

    def get_membership(self, x):
        lower = self.lower_mf(x)
        upper = self.upper_mf(x)
        return lower, upper

# Rule Class
class Rule:
    def __init__(self, antecedents, consequent, op='AND'):
        self.antecedents = antecedents  # list of (input_name, IT2FS)
        self.consequent = consequent    # (output_name, IT2FS)
        self.op = op

    def firing_strength(self, inputs):
        memberships = []
        for name, fs in self.antecedents:
            lower, upper = fs.get_membership(inputs[name])
            memberships.append((lower, upper))
        if self.op == 'AND':
            lower = min(m[0] for m in memberships)
            upper = min(m[1] for m in memberships)
        else:  # OR
            lower = max(m[0] for m in memberships)
            upper = max(m[1] for m in memberships)
        return lower, upper

# IT2FLS Class
class IT2FLS:
    def __init__(self, input_names, output_name, rules, output_universe):
        self.input_names = input_names
        self.output_name = output_name
        self.rules = rules
        self.output_universe = output_universe

    def evaluate(self, inputs):
        fired_outputs = []
        for rule in self.rules:
            firing_lower, firing_upper = rule.firing_strength(inputs)
            out_name, out_fs = rule.consequent
            for out_x in self.output_universe:
                out_lower, out_upper = out_fs.get_membership(out_x)
                result_lower = min(firing_lower, out_lower)
                result_upper = min(firing_upper, out_upper)
                fired_outputs.append((result_lower, result_upper, out_x))
        return fired_outputs

    def defuzzify(self, fired_outputs):
        sum_lower = sum_upper = sum_lower_w = sum_upper_w = 0.0
        for lower, upper, x in fired_outputs:
            sum_lower += lower * x
            sum_upper += upper * x
            sum_lower_w += lower
            sum_upper_w += upper
        centroid_lower = sum_lower / sum_lower_w if sum_lower_w > 0 else 0
        centroid_upper = sum_upper / sum_upper_w if sum_upper_w > 0 else 0
        centroid = (centroid_lower + centroid_upper) / 2
        return centroid_lower, centroid_upper, centroid

# Universe of Discourse
accel_universe = np.linspace(0, 8, 100)
motion_universe = np.linspace(0, 0.2, 100)
risk_universe = np.linspace(0, 100, 100)
pedal_still_universe = np.linspace(0.0, 0.5, 100)
tilt_universe = np.linspace(0, 0.2, 100)
z_orientation_universe = np.linspace(0, 6, 100)

# Fuzzy Sets
lift_low = IT2FS(
    lower_mf=tri_mf(0.71, 0.8, 0.9, 1.0),    
    upper_mf=tri_mf(0.7, 0.85, 0.95, 0.8)
)
lift_normal = IT2FS(
    lower_mf=trap_mf(0.9, 0.99, 1.1, 1.2, 1.0),  
    upper_mf=trap_mf(0.85, 0.99, 1.15, 1.25, 0.8)
)
lift_high = IT2FS(
    lower_mf=trap_mf(1.2, 1.2, 8.0, 9.0, 1.0),    
    upper_mf=trap_mf(1.15, 1.15, 8.0, 9.0, 0.8)
)

z_normal = IT2FS(
    lower_mf=trap_mf(0.9, 0.95, 1.0, 1.02, 1.0),
    upper_mf=trap_mf(0.88, 0.93, 1.01, 1.04, 0.8)
)
z_laid = IT2FS(
    lower_mf=trap_mf(0, 0, 0.45, 0.75, 1.0),
    upper_mf=trap_mf(0, 0, 0.5, 0.78, 0.8)
)
z_lift = IT2FS(
    lower_mf=trap_mf(1.2, 1.3, 5.5, 6.0, 1.0),
    upper_mf=trap_mf(1.15, 1.25, 5.8, 6.2, 0.8)
)

pedal_still_low = IT2FS(
    lower_mf=trap_mf(0, 0, 0.04, 0.06, 1.0),
    upper_mf=trap_mf(0, 0, 0.05, 0.07, 0.8)
)

pedal_still_high = IT2FS(
    lower_mf=trap_mf(0.06, 0.08, 0.5, 0.5, 1.0),
    upper_mf=trap_mf(0.05, 0.07, 0.5, 0.5, 0.8)
)

motion_static = IT2FS(
    lower_mf=trap_mf(0, 0, 0.01, 0.043, 1.0),  
    upper_mf=trap_mf(0, 0, 0.02, 0.0435, 0.8)   
)
motion_moving = IT2FS(
    lower_mf=trap_mf(0.044, 0.044, 0.1, 0.2, 1.0),  
    upper_mf=trap_mf(0.043, 0.04, 0.15, 0.25, 0.8) 
)

tilt_park = IT2FS(
    lower_mf=trap_mf(0, 0, 0.005, 0.0085, 1.0),
    upper_mf=trap_mf(0, 0, 0.004, 0.0085, 0.8)
)
tilt_high = IT2FS(
    lower_mf=trap_mf(0.009, 0.01, 0.03, 0.5, 1.0),
    upper_mf=trap_mf(0.0085, 0.01, 0.04, 0.6, 0.8)
)

risk_low = IT2FS(
    lower_mf=tri_mf(0, 0, 40, 1.0),
    upper_mf=tri_mf(0, 0, 60, 0.8)
)
risk_medium = IT2FS(
    lower_mf=trap_mf(20, 50, 70, 90, 1.0),
    upper_mf=trap_mf(10, 40, 80, 100, 0.8)
)
risk_high = IT2FS(
    lower_mf=tri_mf(60, 100, 100, 1.0),
    upper_mf=tri_mf(40, 100, 100, 0.8)
)

rules = [
    # ... (same rules as before, copy from your original file) ...
    Rule([("tilt", tilt_high), ("z_orientation", z_laid)], ("theft_risk", risk_medium), op='AND'),
    Rule([("tilt", lift_high), ("z_orientation", z_laid)], ("theft_risk", risk_medium), op='AND'),
    Rule([("tilt", lift_high), ("z_orientation", tilt_high)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_high), ("car_motion", motion_moving)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_high), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_laid), ("tilt", tilt_high), ("car_motion", motion_static), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_high), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_laid), ("tilt", tilt_high), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_park), ("car_motion", motion_moving)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_lift), ("tilt", tilt_park), ("car_motion", motion_static), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_park), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_lift), ("tilt", tilt_park), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_normal), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_normal), ("z_orientation", z_lift), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_high)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("z_orientation", z_lift), ("car_motion", motion_static), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_normal), ("z_orientation", z_lift), ("car_motion", motion_moving), ("pedal_stillness", pedal_still_low)], ("theft_risk", risk_high), op='AND'),
    Rule([("lift_accel", lift_normal), ("tilt", tilt_park), ("car_motion", motion_moving)], ("theft_risk", risk_medium), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_high), ("car_motion", motion_static)], ("theft_risk", risk_low), op='AND'),
    Rule([("lift_accel", lift_high), ("tilt", tilt_park), ("car_motion", motion_static)], ("theft_risk", risk_low), op='AND'),
    Rule([("lift_accel", lift_low), ("car_motion", motion_static)], ("theft_risk", risk_low), op='AND'),
    Rule([("lift_accel", lift_low), ("car_motion", motion_moving)], ("theft_risk", risk_low), op='AND'), 
    Rule([("lift_accel", lift_high), ("pedal_stillness", pedal_still_high)], ("theft_risk", risk_low), op='AND'),
    Rule([("lift_accel", lift_normal), ("pedal_stillness", pedal_still_high)], ("theft_risk", risk_low), op='AND'),
]

input_names = [
    "lift_accel", "pedal_stillness", "car_motion", "tilt",
    "z_orientation" 
]
output_name = "theft_risk"
fls = IT2FLS(input_names, output_name, rules, risk_universe)

def print_input_memberships(inputs):
    categories = {
        "lift_accel": [("Low", lift_low), ("Normal", lift_normal), ("High", lift_high)],
        "pedal_stillness": [("Low", pedal_still_low), ("High", pedal_still_high)],
        "car_motion": [("Static", motion_static), ("Moving", motion_moving)],
        "tilt": [("Park", tilt_park), ("High", tilt_high)],
        "z_orientation": [("Normal", z_normal), ("Laid", z_laid), ("Lift", z_lift)],
    }
    print("\nInput Memberships:")
    for key, sets in categories.items():
        val = inputs[key]
        memberships = []
        for name, fs in sets:
            lower, upper = fs.get_membership(val)
            memberships.append(f"{name}: {lower:.2f}-{upper:.2f}")
        print(f"  {key} ({val:.3f}): " + ", ".join(memberships))

# Export relevant objects
__all__ = [
    "fls", "print_input_memberships",
    "accel_universe", "motion_universe", "risk_universe",
    "pedal_still_universe", "tilt_universe", "z_orientation_universe"
]