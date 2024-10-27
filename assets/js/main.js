document.addEventListener('DOMContentLoaded', function() {
    var dropdowns = document.querySelectorAll('.dropdown-toggle');
    
    dropdowns.forEach(function(dropdown) {
        dropdown.addEventListener('click', function(e) {
            e.preventDefault();
            var menu = this.nextElementSibling;
            menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
        });
    });
});