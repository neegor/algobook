import os
import yaml
from glob import glob


def extract_tags_from_md(filepath):
    """Извлекает теги из YAML front matter Markdown-файла."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if content.startswith("---"):
        front_matter = content.split("---")[1]
        metadata = yaml.safe_load(front_matter)
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [
                tag.strip() for tag in tags.split(",")
            ]  # Разбиваем строку по запятым
        return tags
    return []


def generate_nav_by_tags(docs_dir="docs"):
    """Генерирует структуру навигации для mkdocs.yml на основе тегов."""
    md_files = glob(os.path.join(docs_dir, "**/*.md"), recursive=True)
    tags_map = {}

    for filepath in md_files:
        rel_path = os.path.relpath(filepath, docs_dir)
        tags = extract_tags_from_md(filepath)
        if not tags:
            continue
        for tag in tags:
            if tag not in tags_map:
                tags_map[tag] = []
            tags_map[tag].append(rel_path)

    # Сортируем теги по алфавиту
    sorted_tags = sorted(tags_map.keys())

    # Формируем nav: каждый тег -> список файлов (в алфавитном порядке)
    nav = [{tag: tags_map[tag]} for tag in sorted_tags]

    # Добавляем "Другое" для файлов без тегов
    untagged_files = [
        os.path.relpath(f, docs_dir)
        for f in md_files
        if not extract_tags_from_md(f)
        and os.path.relpath(f, docs_dir)
        not in [f for files in tags_map.values() for f in files]
    ]
    if untagged_files:
        nav.append({"Другое": untagged_files})

    return nav


if __name__ == "__main__":
    nav = generate_nav_by_tags()
    print("Навигация на основе тегов (отсортирована по алфавиту):")
    print(
        yaml.dump(
            {"nav": nav}, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
    )
