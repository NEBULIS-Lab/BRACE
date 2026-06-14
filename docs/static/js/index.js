window.HELP_IMPROVE_VIDEOJS = false;

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

// Lazy load videos - only load video source when it enters viewport
function setupLazyVideoLoading() {
    const lazyVideos = document.querySelectorAll('video[data-src]');
    
    if (lazyVideos.length === 0) return;
    
    // Check if IntersectionObserver is supported
    if (typeof IntersectionObserver === 'undefined') {
        // Fallback: load all videos immediately for older browsers
        lazyVideos.forEach(video => {
            const videoSrc = video.getAttribute('data-src');
            if (videoSrc) {
                const source = document.createElement('source');
                source.src = videoSrc;
                source.type = 'video/mp4';
                video.appendChild(source);
                video.removeAttribute('data-src');
                video.load();
            }
        });
        return;
    }
    
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                const videoSrc = video.getAttribute('data-src');
                
                if (videoSrc && !video.querySelector('source')) {
                    // Create source element and add to video
                    const source = document.createElement('source');
                    source.src = videoSrc;
                    source.type = 'video/mp4';
                    video.appendChild(source);
                    
                    // Remove data-src to prevent reloading
                    video.removeAttribute('data-src');
                    
                    // Load the video
                    video.load();
                }
                
                // Stop observing this video
                videoObserver.unobserve(video);
            }
        });
    }, {
        rootMargin: '100px', // Start loading 100px before video enters viewport
        threshold: 0.01 // Trigger even if only 1% is visible
    });
    
    lazyVideos.forEach(video => {
        videoObserver.observe(video);
    });
    
    // Also load videos on click/interaction to ensure they work
    lazyVideos.forEach(video => {
        video.addEventListener('click', function() {
            const videoSrc = video.getAttribute('data-src');
            if (videoSrc && !video.querySelector('source')) {
                const source = document.createElement('source');
                source.src = videoSrc;
                source.type = 'video/mp4';
                video.appendChild(source);
                video.removeAttribute('data-src');
                video.load();
            }
        });
        
        // Also trigger on play attempt
        video.addEventListener('play', function() {
            const videoSrc = video.getAttribute('data-src');
            if (videoSrc && !video.querySelector('source')) {
                const source = document.createElement('source');
                source.src = videoSrc;
                source.type = 'video/mp4';
                video.appendChild(source);
                video.removeAttribute('data-src');
                video.load();
            }
        }, { once: true });
    });
}

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Fix video sizing: ensure poster box matches video box before playback
function fixVideoSizing() {
    const demoVideos = document.querySelectorAll('#demos video');
    demoVideos.forEach(video => {
        // Set size immediately if metadata is already loaded
        if (video.readyState >= 1) {
            video.style.width = video.offsetWidth + 'px';
            video.style.height = video.offsetHeight + 'px';
        }
        
        // Lock size when metadata loads
        video.addEventListener('loadedmetadata', function() {
            const computedWidth = window.getComputedStyle(video).width;
            const computedHeight = window.getComputedStyle(video).height;
            video.style.width = computedWidth;
            video.style.height = computedHeight;
        }, { once: true });
        
        // Also lock on canplay to catch early loads
        video.addEventListener('canplay', function() {
            const computedWidth = window.getComputedStyle(video).width;
            const computedHeight = window.getComputedStyle(video).height;
            video.style.width = computedWidth;
            video.style.height = computedHeight;
        }, { once: true });
    });
}

function setupDeferredYouTubeEmbeds() {
    const embedShells = document.querySelectorAll('.video-embed-shell[data-video-src]');

    embedShells.forEach(shell => {
        const button = shell.querySelector('.video-placeholder-button');
        if (!button) return;

        button.addEventListener('click', function() {
            const videoSrc = shell.getAttribute('data-video-src');
            const videoTitle = shell.getAttribute('data-video-title') || 'Embedded video';
            if (!videoSrc) return;

            const iframe = document.createElement('iframe');
            iframe.src = videoSrc;
            iframe.title = videoTitle;
            iframe.loading = 'lazy';
            iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
            iframe.allowFullscreen = true;

            shell.replaceChildren(iframe);
            shell.removeAttribute('data-video-src');
        }, { once: true });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    setupThemeToggle();
    setupLazyVideoLoading();
    setupVideoCarouselAutoplay();
    fixVideoSizing();
    setupDeferredYouTubeEmbeds();
});
