// Street Voices — public pages JS
// Make Name field required, replace "Powered by listmonk" footer with SV branding

(function () {
  function init() {
    var nameInput = document.getElementById('name');
    if (nameInput) {
      nameInput.required = true;
      nameInput.placeholder = 'Name';
      var label = document.querySelector('label[for="name"]');
      if (label) label.textContent = 'Name';
    }

    var footer = document.querySelector('footer.container');
    if (footer) {
      var year = new Date().getFullYear();
      footer.innerHTML =
        '<div style="text-align:center;font-size:12px;color:#666;line-height:18px;">' +
          '&copy; ' + year + ' Street Voices &nbsp;&middot;&nbsp; ' +
          '<a href="https://streetvoices.ca">streetvoices.ca</a> &nbsp;&middot;&nbsp; ' +
          '720 Bathurst Street, Toronto, ON M5S 2R4' +
        '</div>';
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
