from    rohan.common.logging                   import Logger
from    rohan.common.base_guidances            import GuidanceBase
from    numpy.typing                           import NDArray
from    typing                                 import Union, TypeVar, List, Optional, Dict, Any
from    time                                   import perf_counter
from    math                                   import cos, sin, pi

PRESET_CIRCLE       = '0'
# NOTE: Preset circle will start trajectory at [r, 0]

PRESET_RECTANGLE    = '1'
# NOTE: Preset circle will start trajectory at [l/2, w/2]

SelfPreset2DShapes = TypeVar("SelfPreset2DShapes", bound="Preset2DShapes" )
class Preset2DShapes( GuidanceBase ):

    process_name        : str                   = "Preset Shapes Guidance"

    traj_start_time     : Optional[float]      = None
    shape               : str
    shape_params        : Dict[str,Any]

    def __init__(
        self,
        shape           : str,
        shape_params    : Dict[str,Any]    = {},
        logger          : Optional[Logger] = None, 
        **kwargs,
    ):
        super().__init__(
            logger=logger,
        )

        if shape is PRESET_CIRCLE:
            if not all( key in shape_params for key in ('r','freq') ): 
                raise KeyError
            if 'c_pnt' not in shape_params:
                shape_params['c_pnt'] = (0.,0.)
        elif shape is PRESET_RECTANGLE:
            raise NotImplementedError # TODO:
            if not all( key in shape_params for key in ('l','w','v') ): 
                raise KeyError
        else: 
            raise NotImplementedError

        self.shape          = shape
        self.shape_params   = shape_params
        self.load( **kwargs )
    
    def init_guidance( self ):
        # NOTE: Nothing needs to be loaded at the moment
        pass

    def deinit_guidance( self ):
        # NOTE: Nothing needs to be loaded at the moment
        pass

    def get_guidance_time(self):
        if not self.traj_start_time is None:
            return perf_counter() - self.traj_start_time
        return None
    
    def reset_guidance(self):
        self.traj_start_time = None

    def get_init_guidance(self):
        if self.shape is PRESET_CIRCLE:
            return {
                'x'     : self.shape_params['c_pnt'][0] + self.shape_params['r'], 
                'y'     : 0.,
                'yaw'   : -pi
            }
        elif self.shape is PRESET_RECTANGLE:
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

    def determine_guidance( 
        self,
    ) -> Optional[ Union[ List[float], NDArray ] ]:
    
        if self.traj_start_time is None: self.traj_start_time = perf_counter()
        if self.shape is PRESET_CIRCLE:
            tau     = perf_counter() - self.traj_start_time
            r       = self.shape_params['r']
            freq    = self.shape_params['freq']
            c_pnt   = self.shape_params['c_pnt']
            x       = r *         cos( freq * tau )
            y       = r *         sin( freq * tau )
            v_x     = r * freq * -sin( freq * tau )
            v_y     = r * freq *  cos( freq * tau )
            return {
                'x'     : c_pnt[0] + x   , 
                'y'     : c_pnt[1] + y   ,
                'v_x'   : v_x ,
                'v_y'   : v_y ,
                'yaw'   : -pi + freq * tau,
                'av_z'  : freq
            }
        elif self.shape is PRESET_RECTANGLE:
            raise NotImplementedError
        else: 
            raise NotImplementedError
        
        return None