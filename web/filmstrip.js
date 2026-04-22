// Left-rail filmstrip: thumbnail per page, click-to-select, status chip.

export function initFilmstrip({ root, onSelect }) {
  root.innerHTML = "";
  const state = { pages: [], currentIndex: -1 };

  function render() {
    root.innerHTML = "";
    state.pages.forEach((p, i) => {
      const cell = document.createElement("div");
      cell.className = "filmstrip-cell";
      cell.classList.toggle("filmstrip-current", i === state.currentIndex);
      const img = document.createElement("img");
      img.src = p.thumbnailUrl;
      img.alt = `page ${i + 1}`;
      const chip = document.createElement("span");
      chip.className = `chip chip-${p.status}`;
      chip.title = p.status;
      const badge = document.createElement("span");
      badge.className = "filmstrip-badge";
      badge.textContent = `${i + 1}`;
      cell.append(img, chip, badge);
      cell.addEventListener("click", () => onSelect(i));
      root.append(cell);
    });
  }

  return {
    setPages(pages) {
      state.pages = pages;
      render();
    },
    setCurrent(index) {
      state.currentIndex = index;
      render();
    },
    setStatus(index, status) {
      if (state.pages[index]) {
        state.pages[index].status = status;
        render();
      }
    },
  };
}
