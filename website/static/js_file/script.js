document.getElementById('uploadForm').addEventListener('submit', function(){
    // card ke niche processing message
    document.getElementById('miniLoader').style.display = 'flex';

    // disable button taki baar baar click na ho
    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.style.opacity = '0.7';

    // hide if there is a previous result
    const oldResult = document.getElementById('resultSection');
    if(oldResult){
        oldResult.style.display = 'none';
    }
});