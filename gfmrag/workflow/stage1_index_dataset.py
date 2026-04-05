import logging
import os
import torch
import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import faiss.contrib.torch_utils
from gfmrag import KGIndexer
from gfmrag.kg_construction import KGConstructor, QAConstructor

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

torch.cuda.set_device(1)

if "OPENAI_API_KEY" not in os.environ:
    os.environ['OPENAI_API_KEY'] = 'sk-DUMMYKEYFORLOCALSERVER'


@hydra.main(config_path="config", config_name="stage1_index_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    kg_constructor = KGConstructor.from_config(cfg.kg_constructor)
    qa_constructor = QAConstructor.from_config(cfg.qa_constructor)

    kg_indexer = KGIndexer(kg_constructor, qa_constructor)
    kg_indexer.index_data(cfg.dataset)


if __name__ == "__main__":
    main()
