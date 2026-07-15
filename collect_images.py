"""
icrawler(BingImageCrawler)로 헬스장 기구 12종 이미지를 클래스별로 수집해
dataset/train/<class>/ 에 저장합니다.

사용 예:
    pip install icrawler
    python collect_images.py --num-images 50

주의:
- 검색 결과에는 워터마크/일러스트/무관한 이미지가 섞일 수 있으니
  다운로드 후 각 클래스 폴더를 눈으로 훑어보고 이상한 이미지는 지워주세요.
- 같은 클래스를 다시 실행하면 file_idx_offset="auto" 설정으로
  기존 파일 뒤에 이어서 저장되어 기존 이미지를 덮어쓰지 않습니다.
- 클래스마다 검색어를 여러 개(variant) 등록해두었습니다. 하나의 검색어로는
  Bing이 반환하는 결과 수가 금방 바닥나므로, 폴더 내 이미지 수가 목표치
  (--num-images)에 도달할 때까지 다음 검색어로 넘어가며 이어서 수집합니다.
- Bing SafeSearch(adlt=strict)를 강제 적용하지만 100% 필터링을 보장하지는
  않으므로, 다운로드 후 반드시 각 클래스 폴더를 눈으로 훑어봐야 합니다.
"""

import argparse
import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler
from icrawler.builtin.bing import BingFeeder


class SafeBingFeeder(BingFeeder):
    """BingFeeder에 SafeSearch(adlt=strict)를 강제 적용하는 feeder.

    icrawler의 필터 API는 Bing의 성인 콘텐츠 필터(adlt)를 노출하지 않아서
    쿼리 URL에 직접 파라미터를 추가하도록 feed()를 오버라이드합니다.
    """

    def feed(self, keyword, offset, max_num, filters=None):
        base_url = "https://www.bing.com/images/async?q={}&first={}"
        self.filter = self.get_filter()
        filter_str = self.filter.apply(filters)
        filter_str = "&qft=" + filter_str if filter_str else ""

        for i in range(offset, offset + max_num, 20):
            url = base_url.format(keyword, i) + filter_str + "&adlt=strict"
            self.out_queue.put(url)


# 클래스명 -> 검색 키워드 variant 목록 (하나가 고갈되면 다음 검색어로 이어서 수집)
# "gym equipment"를 붙여 구체적으로 명시할수록 인물/유명인/무관 스톡사진 유입이 줄어듭니다.
SEARCH_QUERIES = {
    "lat_pulldown": [
        "랫풀다운 머신",
        "lat pulldown machine gym equipment",
        "wide grip lat pulldown machine gym equipment",
        "cable lat pulldown tower gym equipment",
    ],
    "cable_row": [
        "케이블 로우 머신",
        "seated cable row machine gym equipment",
        "low pulley cable row machine gym equipment",
        "cable rowing machine gym equipment",
    ],
    "chest_press": [
        "체스트 프레스 머신",
        "seated chest press machine gym equipment",
        "plate loaded chest press machine gym equipment",
        "selectorized chest press machine gym equipment",
    ],
    "shoulder_press": [
        "숄더프레스 머신 헬스장",
        "seated shoulder press machine gym equipment",
        "overhead shoulder press machine gym equipment",
        "plate loaded shoulder press machine gym equipment",
        "selectorized shoulder press machine gym equipment",
    ],
    "pec_deck": [
        "펙덱 머신",
        "pec deck machine",
        "butterfly machine gym",
    ],
    "leg_press": [
        "레그 프레스 머신",
        "45 degree leg press machine gym equipment",
        "horizontal leg press machine gym equipment",
        "plate loaded leg press machine gym equipment",
    ],
    "leg_extension": [
        "레그 익스텐션 머신",
        "seated leg extension machine gym equipment",
        "quad extension machine gym equipment",
        "knee extension machine gym equipment",
    ],
    "leg_curl": [
        "레그 컬 머신",
        "leg curl machine",
        "hamstring curl machine gym",
    ],
    "hip_abduction": [
        "힙 어브덕션 머신",
        "hip abduction machine gym equipment",
        "hip adduction machine gym equipment",
        "inner outer thigh machine gym equipment",
    ],
    "cable_crossover": [
        "케이블 크로스오버 머신",
        "cable crossover station",
        "dual cable machine",
    ],
    "smith_machine": [
        "스미스 머신",
        "smith machine gym",
        "smith rack machine",
    ],
    "assisted_pullup": [
        "어시스티드 풀업 머신",
        "assisted pull up machine gym equipment",
        "assisted dip pull up machine gym equipment",
        "assisted chin up machine gym equipment",
    ],
}


def count_existing(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    return sum(1 for p in output_dir.iterdir() if p.is_file())


def collect_class(queries: list[str], output_dir: Path, target_total: int, min_size: tuple[int, int]):
    output_dir.mkdir(parents=True, exist_ok=True)

    for keyword in queries:
        current = count_existing(output_dir)
        remaining = target_total - current
        if remaining <= 0:
            break

        print(f"    검색어 \"{keyword}\" (현재 {current}장, 목표까지 {remaining}장 필요)")
        crawler = BingImageCrawler(
            feeder_cls=SafeBingFeeder,
            storage={"root_dir": str(output_dir)},
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=keyword,
            max_num=remaining,
            min_size=min_size,
            file_idx_offset="auto",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--num-images", type=int, default=50, help="클래스별 목표 총 이미지 수")
    parser.add_argument(
        "--additional",
        type=int,
        default=None,
        help="지정 시 --num-images 대신 '현재 개수 + N장'을 클래스별 목표로 사용",
    )
    parser.add_argument("--min-width", type=int, default=224)
    parser.add_argument("--min-height", type=int, default=224)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=list(SEARCH_QUERIES.keys()),
        help="수집할 클래스만 선택 (기본: 전체 12종)",
    )
    parser.add_argument("--sleep-sec", type=float, default=2.0, help="클래스 간 대기시간(초)")
    args = parser.parse_args()

    unknown = [c for c in args.classes if c not in SEARCH_QUERIES]
    if unknown:
        raise ValueError(f"알 수 없는 클래스: {unknown}. 사용 가능: {list(SEARCH_QUERIES.keys())}")

    for i, class_name in enumerate(args.classes):
        queries = SEARCH_QUERIES[class_name]
        class_dir = args.output_dir / class_name

        if args.additional is not None:
            target_total = count_existing(class_dir) + args.additional
            print(f"[{i + 1}/{len(args.classes)}] '{class_name}' (현재+{args.additional}장 -> 목표 {target_total}장)")
        else:
            target_total = args.num_images
            print(f"[{i + 1}/{len(args.classes)}] '{class_name}' (목표 {target_total}장)")

        collect_class(
            queries=queries,
            output_dir=class_dir,
            target_total=target_total,
            min_size=(args.min_width, args.min_height),
        )

        if i < len(args.classes) - 1:
            time.sleep(args.sleep_sec)

    print("완료. 각 클래스 폴더를 열어 무관한 이미지가 있으면 직접 삭제해주세요.")


if __name__ == "__main__":
    main()
