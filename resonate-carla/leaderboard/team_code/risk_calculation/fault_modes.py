import enum


class FaultModes(enum.Enum):
    """Enumeration of possible fault modes"""
    NO_FAULT = 0
    CENTER_CAM_BLUR = 1
    LEFT_CAM_BLUR = 2
    RIGHT_CAM_BLUR = 3
    LEFT_RIGHT_CAM_BLUR = 4
    CENTER_RIGHT_CAM_BLUR = 5
    CENTER_LEFT_CAM_BLUR = 6
    ALL_CAM_BLUR = 7
    CENTER_CAM_OCCLUDE = 8
    LEFT_CAM_OCCLUDE = 9
    RIGHT_CAM_OCCLUDE = 10
    LEFT_RIGHT_CAM_OCCLUDE = 11
    CENTER_RIGHT_CAM_OCCLUDE = 12
    CENTER_LEFT_CAM_OCCLUDE = 13
    ALL_CAM_OCCLUDE = 14
    RADAR_FAILURE = 15

class SingularFaultModes(enum.Enum):
    """Enumeration of possible fault modes"""
    NO_FAULT = 0
    CENTER_CAM_BLUR = 1
    LEFT_CAM_BLUR = 2
    RIGHT_CAM_BLUR = 3
    CENTER_CAM_OCCLUDE = 4
    LEFT_CAM_OCCLUDE = 5
    RIGHT_CAM_OCCLUDE = 6
    RADAR_FAILURE = 7


def fault_mode_to_set(fault_mode):
    FM = FaultModes
    SFM = SingularFaultModes
    if fault_mode == 0:
        return [SFM.NO_FAULT]
    if fault_mode == 1:
        return [SFM.CENTER_CAM_BLUR]
    if fault_mode == 2:
        return [SFM.LEFT_CAM_BLUR]
    if fault_mode == 3:
        return [SFM.RIGHT_CAM_BLUR]
    if fault_mode == 4:
        return [SFM.LEFT_CAM_BLUR, SFM.RIGHT_CAM_BLUR]
    if fault_mode == 5:
        return [SFM.CENTER_CAM_BLUR, SFM.RIGHT_CAM_BLUR]
    if fault_mode == 6:
        return [SFM.LEFT_CAM_BLUR, SFM.CENTER_CAM_BLUR]
    if fault_mode == 7:
        return [SFM.CENTER_CAM_BLUR, SFM.LEFT_CAM_BLUR, SFM.RIGHT_CAM_BLUR]
    if fault_mode == 8:
        return [SFM.CENTER_CAM_OCCLUDE]
    if fault_mode == 9:
        return [SFM.LEFT_CAM_OCCLUDE]
    if fault_mode == 10:
        return [SFM.RIGHT_CAM_OCCLUDE]
    if fault_mode == 11:
        return [SFM.LEFT_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE]
    if fault_mode == 12:
        return [SFM.CENTER_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE]
    if fault_mode == 13:
        return [SFM.CENTER_CAM_OCCLUDE, SFM.LEFT_CAM_OCCLUDE]
    if fault_mode == 14:
        return [SFM.CENTER_CAM_OCCLUDE, SFM.LEFT_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE]
    if fault_mode == 15:
        return [SFM.RADAR_FAILURE]


    # map = {
    #     FM.NO_FAULT: [SFM.NO_FAULT],
    #     FM.CENTER_CAM_BLUR: [SFM.CENTER_CAM_BLUR],
    #     FM.LEFT_CAM_BLUR: [SFM.LEFT_CAM_BLUR],
    #     FM.RIGHT_CAM_BLUR: [SFM.RIGHT_CAM_BLUR],
    #     FM.LEFT_RIGHT_CAM_BLUR: [SFM.LEFT_CAM_BLUR, SFM.RIGHT_CAM_BLUR],
    #     FM.CENTER_RIGHT_CAM_BLUR: [SFM.CENTER_CAM_BLUR, SFM.RIGHT_CAM_BLUR],
    #     FM.CENTER_LEFT_CAM_BLUR: [SFM.LEFT_CAM_BLUR, SFM.CENTER_CAM_BLUR],
    #     FM.ALL_CAM_BLUR: [SFM.CENTER_CAM_BLUR, SFM.LEFT_CAM_BLUR, SFM.RIGHT_CAM_BLUR],
    #     FM.CENTER_CAM_OCCLUDE: [SFM.CENTER_CAM_OCCLUDE],
    #     FM.LEFT_CAM_OCCLUDE: [SFM.LEFT_CAM_OCCLUDE],
    #     FM.RIGHT_CAM_OCCLUDE: [SFM.RIGHT_CAM_OCCLUDE],
    #     FM.LEFT_RIGHT_CAM_OCCLUDE: [SFM.LEFT_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE],
    #     FM.CENTER_RIGHT_CAM_OCCLUDE: [SFM.CENTER_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE],
    #     FM.CENTER_LEFT_CAM_OCCLUDE: [SFM.CENTER_CAM_OCCLUDE, SFM.LEFT_CAM_OCCLUDE],
    #     FM.ALL_CAM_OCCLUDE: [SFM.CENTER_CAM_OCCLUDE, SFM.LEFT_CAM_OCCLUDE, SFM.RIGHT_CAM_OCCLUDE],
    #     FM.RADAR_FAILURE: [SFM.RADAR_FAILURE]
    # }
    #print(map[fault_mode])
    #return map[fault_mode]
