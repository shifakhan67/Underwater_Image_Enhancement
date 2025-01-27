document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('.section');

    function hideAllSections() {
        sections.forEach(section => {
            section.style.display = 'none';
        });
    }

    function showSection(sectionId) {
        hideAllSections();
        const section = document.getElementById(`${sectionId}-section`);
        if (section) {
            section.style.display = 'block';
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionToShow = link.getAttribute('data-section');
            showSection(sectionToShow);
        });
    });

    // Ensure home section is visible by default
    showSection('home');
});