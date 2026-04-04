/**
 * Portfolio site behavior:
 * - Mobile nav open/close + sync aria-expanded for assistive tech
 * - Close menu when a nav link is clicked (mobile UX)
 * - Current year in footer
 * - Scroll-triggered reveals: .is-visible on main > section.section and on
 *   .site-footer; disabled when user prefers reduced motion (shown at once).
 */

(function () {
  const nav = document.getElementById("site-nav");
  const toggle = document.querySelector(".nav-toggle");
  const navLinks = nav ? nav.querySelectorAll("a[href^='#']") : [];

  const yearEl = document.getElementById("year");
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }

  // --- Scroll reveals: IntersectionObserver adds .is-visible once per section ---
  const revealSections = document.querySelectorAll("main > section.section");
  const prefersReducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function markAllSectionsVisible() {
    revealSections.forEach(function (section) {
      section.classList.add("is-visible");
    });
  }

  if (revealSections.length) {
    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      // No motion / no API: show content immediately (matches CSS fallback)
      markAllSectionsVisible();
    } else {
      const io = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-visible");
              io.unobserve(entry.target);
            }
          });
        },
        {
          root: null,
          // Fire a bit before the section fully enters (feels more natural)
          rootMargin: "0px 0px -10% 0px",
          threshold: 0.08,
        }
      );
      revealSections.forEach(function (section) {
        io.observe(section);
      });
    }
  }

  /* Footer: same reveal idea as main sections (element lives outside <main>) */
  const footerEl = document.querySelector(".site-footer");
  if (footerEl) {
    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      footerEl.classList.add("is-visible");
    } else {
      const footerIo = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              entry.target.classList.add("is-visible");
              footerIo.unobserve(entry.target);
            }
          });
        },
        { root: null, rootMargin: "0px", threshold: 0.12 }
      );
      footerIo.observe(footerEl);
    }
  }

  if (!nav || !toggle) return;

  function setOpen(isOpen) {
    nav.classList.toggle("is-open", isOpen);
    toggle.setAttribute("aria-expanded", String(isOpen));
    toggle.setAttribute(
      "aria-label",
      isOpen ? "Close menu" : "Open menu"
    );
  }

  toggle.addEventListener("click", function () {
    const willOpen = !nav.classList.contains("is-open");
    setOpen(willOpen);
  });

  navLinks.forEach(function (link) {
    link.addEventListener("click", function () {
      if (window.matchMedia("(max-width: 768px)").matches) {
        setOpen(false);
      }
    });
  });

  window.addEventListener("resize", function () {
    if (!window.matchMedia("(max-width: 768px)").matches) {
      setOpen(false);
    }
  });
})();
