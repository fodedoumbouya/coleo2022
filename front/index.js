localStorage.setItem("infos", "{}");
localStorage.setItem("clicked-point", "{}");
var classJson = [
    {
        "classe":"coleonica",
        "couleur":"#87CEEB"
    },
    {
        "classe":"faux",
        "couleur":"#B0E0E6"
    },
    {
        "classe":"flat",
        "couleur":"#B0C4DE"
    },
    {
        "classe":"placodea",
        "couleur":"#4682B4"
    },
    {
        "classe":"pore",
        "couleur":"#FFE4C4"
    },
    {
        "classe":"ringed",
        "couleur":"#e7feff"
    },
    {
        "classe":"roundtip",
        "couleur":"#F0E68C"
    },
    {
        "classe":"sting",
        "couleur":"#a1b19f"
    },
    {
        "classe":"wrinkled",
        "couleur":"#9ACD32"
    },
    {
        "classe":"sunken",
        "couleur":"#bcd4e6"
    }
]
function generateClass(){
    let classHtml=""
    classJson.forEach(eachClass => {
        classHtml +=`<div id="${eachClass.classe}" class="image-item" style="background-color: ${eachClass.couleur};">
                        ${eachClass.classe}
                    </div>`
    })
    $("#mouche-classe").html(classHtml)
    let allClass=document.getElementsByClassName("image-item")
    if(allClass && allClass.length > 0){
        let allClassLength=allClass.length
        for (let index = 0; index < allClassLength; index++) {
            const element = allClass[index];
            element.addEventListener("click", (e)=>{
                $(".image-item").css("border-color","black")
                allClass[index].style.borderColor  ="red"
                updatePoint(JSON.parse(localStorage.getItem("clicked-point")),allClass[index],id);
            })
        }
    }
}

function updatePoint(coordinates=[],label=""){
   let infos= JSON.parse(localStorage.getItem("infos"));
   if(coordinates.length>0){
        infos[coordinates.join("-")]={
            coordinates:coordinates,
            label:label
        }
        infos=JSON.stringify(infos)
        localStorage.setItem("infos", infos);
   }
}

generateClass()