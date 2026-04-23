(() => {
    const storageKey = "theme";
    const root = document.documentElement;
    const themeToggle = document.querySelector("[data-theme-toggle]");
    const fileInput = document.querySelector("[data-file-input]");
    const fileName = document.querySelector("[data-file-name]");

    const applyTheme = (theme) => {
        root.dataset.theme = theme;

        if (!themeToggle) {
            return;
        }

        themeToggle.textContent = theme === "dark" ? "Светлая тема" : "Тёмная тема";
    };

    const currentTheme = root.dataset.theme || "light";
    applyTheme(currentTheme);

    if (themeToggle) {
        themeToggle.addEventListener("click", () => {
            const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
            localStorage.setItem(storageKey, nextTheme);
            applyTheme(nextTheme);
        });
    }

    if (!fileInput || !fileName) {
        return;
    }

    fileInput.addEventListener("change", () => {
        const [file] = fileInput.files ?? [];
        fileName.textContent = file ? file.name : "Файл не выбран";
    });
})();
