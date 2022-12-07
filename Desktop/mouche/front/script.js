
function myFile()
{
    document.getElementById('file').click();
}

function get_coordinates(event)
{
    click_coordinates = {
        X: `${event.clientX}`,
        Y: `${event.clientY}`
    }
    return click_coordinates
}

document.getElementById('file').onchange = function () {
    let file = this.value.replace('C:\\fakepath\\', ' ')
    image_div = document.getElementsByClassName('image')[0]
    image = `<img id="image" src="${file}" />`
     
    document.getElementsByClassName('image')[0].innerHTML = image 
    document.getElementById('image').addEventListener('click', (event)=>{
        get_coordinates(event)
    })
    console.log(file);
  };


//function that returns the coordinates for the given clicked point in the image div element 
var myImage =document.getElementsByClassName("image")



myImage[0].addEventListener('click', (event) => get_coordinates(event));

openImage = document.getElementsByClassName("open_image")[0]
openImage.addEventListener("click",(event) => {

})