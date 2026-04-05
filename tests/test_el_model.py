def test_colbert_el_model() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.entity_linking_model.ColbertELModel",
            "model_name_or_path": "colbert-ir/colbertv2.0",
            "root": "tmp",
            "force": False,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]

    entity_list = [
        "controversy surrounding chief illiniwek",
        "supervisor in the state s attorney s office",
        "may 31  2016",
        "trial of john wayne gacy",
        "june 4  1931",
        "former cook county judge",
        "louis b  garippo",
        "trial of richard speck",
        "richard speck",
        "december 5  1991",
        "eight student nurses",
        "july 13 14  1966",
        "american mass murderer",
        "south chicago community hospital",
        "december 6  1941",
        "beaulieu mine",
        "northwest territories",
        "930 g",
        "yellowknife",
        "7 troy ounces",
        "chaos and bankruptcy",
        "november",
        "world war ii",
        "30 troy ounces",
        "october 1947",
        "1948",
        "schumacher",
        "porcupine gold rush",
        "downtown timmins",
        "mcintyre mine",
        "abandoned underground gold mine",
        "canada",
        "ontario",
        "canadian mining history",
        "the nation s most important mines",
        "headframe",
        "considerable amount of copper",
    ]
    el_model.index(entity_list)
    linked_entity_dict = el_model(ner_entity_list, topk=2)
    print(linked_entity_dict)
    assert isinstance(linked_entity_dict, dict)


def test_dpr_el_model() -> None:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "_target_": "gfmrag.kg_construction.entity_linking_model.DPRELModel",
            "model_name": "BAAI/bge-large-en-v1.5",
            "root": "tmp",
            "use_cache": True,
            "normalize": True,
        }
    )

    el_model = instantiate(cfg)
    ner_entity_list = ["south chicago community hospital", "july 13 14  1966"]

    entity_list = [
        "controversy surrounding chief illiniwek",
        "supervisor in the state s attorney s office",
        "may 31  2016",
        "trial of john wayne gacy",
        "june 4  1931",
        "former cook county judge",
        "louis b  garippo",
        "trial of richard speck",
        "richard speck",
        "december 5  1991",
        "eight student nurses",
        "july 13 14  1966",
        "american mass murderer",
        "south chicago community hospital",
        "december 6  1941",
        "beaulieu mine",
        "northwest territories",
        "930 g",
        "yellowknife",
        "7 troy ounces",
        "chaos and bankruptcy",
        "november",
        "world war ii",
        "30 troy ounces",
        "october 1947",
        "1948",
        "schumacher",
        "porcupine gold rush",
        "downtown timmins",
        "mcintyre mine",
        "abandoned underground gold mine",
        "canada",
        "ontario",
        "canadian mining history",
        "the nation s most important mines",
        "headframe",
        "considerable amount of copper",
    ]
    el_model.index(entity_list)
    linked_entity_list = el_model(ner_entity_list, topk=2)
    print(linked_entity_list)
    assert isinstance(linked_entity_list, dict)


if __name__ == "__main__":
    test_colbert_el_model()
    # test_dpr_el_model()
