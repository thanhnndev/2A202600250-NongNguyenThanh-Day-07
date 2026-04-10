"""Benchmark queries for Phase 2 medical product evaluation."""

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "laser eye surgery technology",
        "query_vi": "công nghệ phẫu thuật laser mắt",
        "expected_brand": "schwind",
        "expected_keywords": ["SmartSight", "ATOS", "SmartSurfACE"],
        "expected_keywords_vi": ["giác mạc", "laser", "phẫu thuật"],
        "category": "technology",
    },
    {
        "id": 2,
        "query": "medical sterilization equipment washer disinfector",
        "query_vi": "thiết bị tiệt trùng máy rửa khử trùng",
        "expected_brand": "melag",
        "expected_keywords": ["MELAtherm", "washer", "disinfector"],
        "expected_keywords_vi": ["tiệt trùng", "rửa", "MELAtherm"],
        "category": "equipment",
    },
    {
        "id": 3,
        "query": "phaco equipment ophthalmic surgery cryo",
        "query_vi": "thiết bị phaco phẫu thuật nhãn khoa",
        "expected_brand": "bvi",
        "expected_keywords": ["phaco", "CRYO", "ophthalmic"],
        "expected_keywords_vi": ["phaco", "nhãn khoa", "phẫu thuật"],
        "category": "devices",
    },
    {
        "id": 4,
        "query": "Bowie Dick sterilization control test",
        "query_vi": "kiểm soát tiệt trùng Bowie Dick",
        "expected_brand": "melag",
        "expected_keywords": ["MELAcontrol", "Bowie", "Dick"],
        "expected_keywords_vi": ["MELAcontrol", "tiệt trùng", "kiểm tra"],
        "category": "control_systems",
    },
    {
        "id": 5,
        "query": "cornea treatment therapeutic solutions",
        "query_vi": "điều trị giác mạc giải pháp điều trị",
        "expected_brand": "schwind",
        "expected_keywords": ["TheraCare", "cornea", "therapeutic"],
        "expected_keywords_vi": ["TheraCare", "giác mạc", "điều trị"],
        "category": "therapy",
    },
]
