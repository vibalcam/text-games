var displayArea;
var dialogue;
var dialogueIterator;
var optNum = 0;

function step() {
  // Steps until an options result
  while(true) {
    var iter = dialogueIterator.next()
    if (iter.done) {
      break;
    }

    var result = iter.value;
    if (result instanceof bondage.OptionResult) {
      showOptions(result);
      break;
    } else {
      displayArea.innerHTML += result.text + '<br/>';
    }
  }
}

function compile() {
  displayArea.innerHTML = '';

  dialogue = new bondage.Runner();
  var data = JSON.parse(document.getElementById('yarn').value);
  dialogue.load(data);
  dialogueIterator = dialogue.run('Start');
  step();
}

function showOptions(result) {
  displayArea.innerHTML += '<br/>';
  for (var i = 0; i < result.options.length; i++) {
    displayArea.innerHTML += '<input name="opt-' + optNum + '" type="radio" value="' + i + '">' + result.options[i] + '</input><br/>';
  }
  displayArea.innerHTML += '<input type="button" id="option-button-' + optNum + '" value="Choose"/>'
  displayArea.innerHTML += '<br/><br/>';

  var button = document.getElementById('option-button-' + optNum);
  button.onclick = function () {
    var radios = document.getElementsByName('opt-' + optNum);
    for (var n in radios) {
      var radio = radios[n];
      if (radio.checked) {
        result.select(radio.value);
        optNum++;
        step();
        return;
      }
    }

    console.error('Need to choose an option first!');
  }
}

window.onload = function () {
  displayArea = document.getElementById('display-area');
  compile();
};
