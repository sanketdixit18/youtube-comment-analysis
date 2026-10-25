/* Runs before the page is painted so the saved theme is applied without a flash. */
(function () {
  var theme = null;
  try {
    theme = localStorage.getItem("yt-sentiment-theme");
  } catch (e) { /* storage can be blocked (private mode) - fall back to the system theme */ }
  if (theme !== "dark" && theme !== "light") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  document.documentElement.setAttribute("data-theme", theme);
})();
