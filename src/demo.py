import hydra
import os
from omegaconf import DictConfig



@hydra.main(version_base=None, 
            # config_path="../configs", 
            config_path=os.path.join(os.path.dirname(__file__), "../configs"),
            config_name="general_config")
def main(config:DictConfig):
    
    print("inside")
    
    
    
if __name__ == "__main__":
    main()
    print('done')