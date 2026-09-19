function applyTheme(theme) {
    const isDark = theme === 'dark';
    if (isDark) {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }

    const toggle = document.querySelector('.theme-toggle');
    if (toggle) {
        toggle.setAttribute('aria-pressed', String(isDark));
        toggle.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
    }

    const logo = document.querySelector('.publication-logo');
    if (logo) {
        const lightSrc = logo.getAttribute('data-light-src');
        const darkSrc = logo.getAttribute('data-dark-src');
        logo.src = isDark && darkSrc ? darkSrc : lightSrc;
    }

    const themeMeta = document.querySelector('meta[name="theme-color"]');
    if (themeMeta) {
        themeMeta.setAttribute('content', isDark ? '#000000' : '#0b1f4d');
    }
}

function setupThemeToggle() {
    const toggle = document.querySelector('.theme-toggle');
    if (!toggle) return;

    let savedTheme = 'light';
    try {
        savedTheme = localStorage.getItem('brace-theme') || 'light';
    } catch (error) {
        savedTheme = 'light';
    }

    applyTheme(savedTheme);
    window.addEventListener('storage', function(event) {
        if (event.key === 'brace-theme') applyTheme(event.newValue === 'dark' ? 'dark' : 'light');
    });

    toggle.addEventListener('click', function() {
        const nextTheme = document.documentElement.hasAttribute('data-theme') ? 'light' : 'dark';
        try {
            localStorage.setItem('brace-theme', nextTheme);
        } catch (error) {
            // Theme still changes for the current page even without storage.
        }
        applyTheme(nextTheme);
    });
}

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        if (dropdown && button) {
            dropdown.classList.remove('show');
            button.classList.remove('active');
        }
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Same-origin animation embed: resize to the scene and pause when offscreen.
function setupAnimationEmbed() {
    const frame = document.getElementById('brace-animation');
    if (!frame) return;
    window.addEventListener('message', function(event) {
        if (event.origin !== location.origin || event.source !== frame.contentWindow) return;
        if (event.data?.type !== 'brace-animation-height') return;
        const height = Number(event.data.height);
        if (Number.isFinite(height) && height >= 300 && height <= 4000) frame.style.height = Math.ceil(height) + 'px';
    });
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver(function(entries) {
            document.body.classList.toggle('animation-in-view', entries[0].isIntersecting);
            if (!entries[0].isIntersecting) frame.contentWindow?.postMessage({ type: 'brace-animation-pause' }, location.origin);
        });
        observer.observe(frame);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    setupThemeToggle();
    setupAnimationEmbed();
});
