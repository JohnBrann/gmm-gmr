from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
    new_site,
    string_to_array,
    xml_path_completion,
)

class SortArena(TableArena):
    def __init__(self, colors=None):
        if colors is None:
            self.colors = [
                (1.0, 0.0, 0.0, 0.8),
                (0.0, 1.0, 0.0, 0.8),
                (0.0, 0.0, 1.0, 0.8)
            ]
        else:
            self.colors = colors
        self.plate_size = 0.05 if len(self.colors) > 8 else 0.4 / len(self.colors) if len(self.colors) > 2 else 0.15
        print(self.plate_size)
        self.next_position = [-0.4 + self.plate_size, -0.4 + self.plate_size, 0.0375]
        super().__init__()
    
    def create_plate(self, color):
        # Create body for colored plate, add to worldbody
        plate_body = new_body(name=f"plate-{color}", pos=self.next_position)
        self.worldbody.find("./body[@name='table']").append(plate_body)
        
        # Plate geom attributes
        plate_attribs = {
            "pos": (0, 0, 0),
            "size": (self.plate_size * 0.95, 0.005),
            "type": "cylinder"
        }
        
        collision = new_geom(name=f"plate-{color}_collision", group=0, friction=(1, 0.005, 0.0001), **plate_attribs)
        visual = new_geom(name=f"plate-{color}_visual", group=1, conaffinity=0, contype=0, rgba=color, **plate_attribs)
        plate_body.append(collision)
        plate_body.append(visual)
    
    def _postprocess_arena(self):
        min_coord = -0.4 + self.plate_size
        max_coord = 0.4 - self.plate_size
        for color in self.colors:
            self.create_plate(color)
            if self.next_position[0] + self.plate_size > max_coord:
                if self.next_position[1] + self.plate_size > -0.05:
                    print("Ran out of space for new plates! (This isn't supposed to happen)")
                else:
                    self.next_position[0] = min_coord
                    self.next_position[1] += self.plate_size * 2
            else:
                self.next_position[0] += self.plate_size * 2
