/*  ==========================================
    SHOW UPLOADED IMAGE
* ========================================== */
function readURL1(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
 
        reader.onload = function (e) {
            $('#imageResult1')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
 
$(function () {
    $('#upload1').on('change', function () {
        readURL1(input1);
    });
});
function readURL2(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
 
        reader.onload = function (e) {
            $('#imageResult2')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
 
$(function () {
    $('#upload2').on('change', function () {
        readURL2(input2);
    });
});
 
/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input1 = document.getElementById( 'upload1' );
var infoArea1 = document.getElementById( 'upload-label1' );
 
input1.addEventListener( 'change', showFileName1 );
function showFileName1( event ) {
  var input1 = event.srcElement;
  var fileName1 = input1.files[0].name;
  infoArea1.textContent = 'File name: ' + fileName1;
}
var input2 = document.getElementById( 'upload2' );
var infoArea2 = document.getElementById( 'upload-label2' );
 
input2.addEventListener( 'change', showFileName2 );
function showFileName2( event ) {
  var input2 = event2.srcElement;
  var fileName2 = input2.files[0].name;
  infoArea2.textContent = 'File name: ' + fileName2;
}
