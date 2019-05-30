$(document).ready(function(){

    $("#regressionSubmit").click(function() {	
    $("#regressionSettingsModal").modal("hide");	
    });	
    $("#naiveBayesSubmit").click(function() {	
    $("#naiveBayesSettingsModal").modal("hide");	
    });	
    $("#kmeansSubmit").click(function() {	
    $("#kmeansSettingsModal").modal("hide");	
    });	

  var containers = $(".drag-container").toArray();
  dragula(containers, {
      isContainer: function (e) {
          return e.classList.contains('drag-container');
      }
  });

  $("[type=file]").on("change", function(){
    var file = this.files[0].name;
    $(this).next().text(file);
  });
  
  eel.columns()(function(c){
    var columnArray = c;
    var arrayLength = columnArray.length;
    var elements = document.getElementsByClassName("dataColumns");
    for (var n = 0; n < elements.length; n++) {
      for (var i = 0; i < arrayLength; i++) {
        var item = document.createElement("div");
        item.className = "item";
        item.id = "r"+i;
        var content = document.createElement("span");
        content.innerHTML = columnArray[i];
        item.appendChild(content);
        elements[n].appendChild(item);
      }
    }
  });

});

function updateDf(){
  eel.update_df()
  reinitialize_func()
}



function showTable(){
  eel.table()(function(ret){
    document.getElementById("csvTable").innerHTML = ret;
    return ret
  });
  var contents = document.getElementsByClassName("tableShow");
  var i;
  for (i = 0; i < contents.length; i++) {
    contents[i].style.display = "block";
  }
}

function reinitialize_func(){
  location.reload();
}

function clickImported(){
  var file = document.getElementById("file").files[0];
  Papa.parse(file, {
    complete: function(results) {
      $( "#csvTable" ).empty();
      $( "#csvTable" ).append( '<div class="text-center"><span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span></div>' );
        eel.csvUpload(results.data)(function(ret){
          document.getElementById("csvTable").innerHTML = ret;
          return ret
        });
        //document.getElementById("tableShow").style.display = "block";
    }
  });
}
function regressionDisplay(){
  document.getElementById("regressionButton").className = "processButtonSelected";
  document.getElementById("kmeansButton").className = "processButton";
  document.getElementById("naiveBayesButton").className = "processButton";
}

function naiveBayesDisplay(){
  document.getElementById("naiveBayesButton").className = "processButtonSelected";
  document.getElementById("regressionButton").className = "processButton";
  document.getElementById("kmeansButton").className = "processButton";
}

function kmeansDisplay(){
  document.getElementById("kmeansButton").className = "processButtonSelected";
  document.getElementById("regressionButton").className = "processButton";
  document.getElementById("naiveBayesButton").className = "processButton";
}

function regressionResultsDisplay(){
  var regressionICount = document.getElementById("regressionIndependent").childElementCount;
  var regressionDCount = document.getElementById("regressionDependent").childElementCount;
  if(regressionICount <= 0 && regressionDCount <= 0){
    alert("You did not select a dependent and independent variable.");
    location.reload();
  }else if(regressionICount <= 0){
    alert("You did not select an independent variable.");
    location.reload();
  }else if(regressionDCount <= 0){
    alert("You did not select a dependent variable.")
    location.reload();
  }else if(regressionDCount > 1){
    alert("You should only select one dependent variable.");
    location.reload();
  }
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("lineGraphButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("rdisplay").style.display = "block";
  if(document.getElementById("linearRegression").checked){
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("regressionLabel").innerHTML = "Linear Regression";
    if(regressionICount == 1){
      var dv = document.getElementById("regressionDependent").getElementsByTagName("span")[0].innerHTML;
      var idv = document.getElementById("regressionIndependent").getElementsByTagName("span")[0].innerHTML;
      eel.lin_regression(dv, idv)(function(ret){
        document.getElementById("rdisplay").srcdoc = ret;
      });
      eel.lin_rtable(dv, idv)(function(ret){
        document.getElementById("regressionTable").innerHTML = ret;
      });
    }else if(regressionICount >= 2){
      var dv = document.getElementById("regressionDependent").getElementsByTagName("span")[0].innerHTML;
      document.getElementById("rdisplay").style.display = "none";
      var idv = [];
      var idvs = document.getElementById("regressionIndependent");
      var nidvs = idvs.getElementsByTagName("span");
      for(i=0;i<nidvs.length;i++){
        idv.push(nidvs[i].innerHTML);
      }
      eel.lin_rtable_multi(dv, idv)(function(ret){
        document.getElementById("regressionTable").innerHTML = ret;
      });
    }
  }else if(document.getElementById("polynomialRegression").checked){
    if(regressionDCount > 1 && regressionICount > 1){
      alert("You should only select one dependent variable and one independent variable.");
      location.reload();
    }else if(regressionICount > 1){
      alert("You should only select one independent variable");
      location.reload();
    }
    document.getElementById("regression-rightbar-content").style.display = "block";
    document.getElementById("regressionLabel").innerHTML = "Polynomial Regression";
    var dv = document.getElementById("regressionDependent").getElementsByTagName("span")[0].innerHTML;
    var idv = document.getElementById("regressionIndependent").getElementsByTagName("span")[0].innerHTML;
    eel.poly_regression(dv, idv)(function(ret){
      document.getElementById("rdisplay").srcdoc = ret;
    });
    eel.poly_rtable(dv, idv)(function(ret){
      document.getElementById("regressionTable").innerHTML = ret;
    });
  }else if(document.getElementById("linearRegression").checked == false && document.getElementById("polynomialRegression").checked == false){
    alert("You did not choose the type of regression");
    location.reload();
  }
}

function naiveBayesResultsDisplay(){
  var naiveBayesTCount = document.getElementById("naiveBayesTarget").childElementCount;
  var naiveBayesFCount = document.getElementById("naiveBayesFeatures").childElementCount;
  if(naiveBayesTCount <= 0 && naiveBayesFCount <= 0){
    alert("You did not select a target feature along with other features.");
    location.reload();
  }else if(naiveBayesTCount <= 0){
    alert("You did not select a target feature.");
    location.reload();
  }else if(naiveBayesFCount <= 0){
    alert("You did not select other features.");
    location.reload();
  }else if(naiveBayesTCount > 1){
    alert("You can should only select one target feature.");
    location.reload();
  }
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "block";
  document.getElementById("kmeans-rightbar-content").style.display = "none";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  document.getElementById("confusionButton").style.backgroundColor = "#E3E6E6";
  var ny = document.getElementById("naiveBayesTarget").getElementsByTagName("span")[0].innerHTML;
  var nX = [];
  var xs = document.getElementById("naiveBayesFeatures");
  var nxs = xs.getElementsByTagName("span");
  for(i=0;i<nxs.length;i++){
    nX.push(nxs[i].innerHTML);
  }
  eel.naive_matrix(nX, ny)(function(ret){
    document.getElementById("ndisplay").srcdoc = ret;
  });
  eel.naive_classify(nX, ny)(function(ret){
    document.getElementById("naiveTable").innerHTML = ret;
  });
}

var c = '';
var kdf = [];

function kmeansResultsDisplay(){
  var kmeansFeaturesCount = document.getElementById("kmeansFeatures").childElementCount;
  if(kmeansFeaturesCount <= 0){
    alert("You did not select any feature.");
    location.reload();
  }else if(kmeansFeaturesCount == 1){
    alert("You only selected one feature, please select two or more.");
    location.reload();
  }
  document.getElementById("regression-rightbar-content").style.display = "none";
  document.getElementById("naiveBayes-rightbar-content").style.display = "none";
  document.getElementById("kmeans-rightbar-content").style.display = "block";
  document.getElementById("graphTemp").style.display = "none";
  document.getElementById("regressionLabel").innerHTML = "Regression";
  c = document.getElementById("clusterNumber").value;
  if(c == ""){
    alert("You did not specify the number of clusters.")
    location.reload();
  }
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_centroids(kdf, c)(function(ret){
    document.getElementById("centroidValues").innerHTML = ret;
  });
  eel.kmeans_sil_coef(kdf, c)(function(ret){
    document.getElementById("silhouetteCoefficient").innerHTML = ret;
  });
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  eel.kmeans_centroid_chart(kdf, c)(function(ret){
    document.getElementById("kdisplay").srcdoc = ret;
  });
}

function cluster() {
  document.getElementById("clusterButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("centroidButton").style.backgroundColor = "#FFFFFF";
  var c = document.getElementById("clusterNumber").value;
  var kdf = [];
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_cluster_graph(kdf, c)(function(ret){
    document.getElementById("kdisplay").srcdoc = ret;
  });
}

function centroid() {
  document.getElementById("centroidButton").style.backgroundColor = "#E3E6E6";
  document.getElementById("clusterButton").style.backgroundColor = "#FFFFFF";
  document.getElementById("graphTemp").src = "img/centroidChart.png";
  var c = document.getElementById("clusterNumber").value;
  var kdf = [];
  var kf = document.getElementById("kmeansFeatures");
  var nkf = kf.getElementsByTagName("span");
  for(i=0;i<nkf.length;i++){
    kdf.push(nkf[i].innerHTML);
  }
  eel.kmeans_centroid_chart(kdf, c)(function(ret){
    document.getElementById("kdisplay").srcdoc = ret;
  });
}
/*
function compare() {
  document.getElementById("graphTemp").src = "img/centroidChart.png";
}
*/
