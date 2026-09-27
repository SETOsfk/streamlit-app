# Shiny version of the panel.   shiny::runApp("R")
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(Sys.setlocale("LC_CTYPE", "C.UTF-8"))  # Turkish labels
library(shiny)
library(bslib)
library(jsonlite)

root <- if (file.exists("Consumer_Shopping_Trends_2026.csv")) "." else ".."
d <- read.csv(file.path(root, "Consumer_Shopping_Trends_2026.csv"), stringsAsFactors = TRUE)
A <- fromJSON(file.path(root, "results", "audit.json"))
num <- setdiff(names(d), c("gender", "city_tier", "shopping_preference"))
d$shopping_preference <- factor(d$shopping_preference, c("Store", "Online", "Hybrid"))

ui <- page_navbar(
  title = "Consumer shopping panel",
  theme = bs_theme(version = 5, primary = "#2d6a8f", base_font = "IBM Plex Sans, system-ui"),
  sidebar = sidebar(radioButtons("lang", NULL, setNames(c("tr", "en"), c("Türkçe", "English")), inline = TRUE),
                    selectInput("col", "Sütun / Column", num, selected = "need_touch_feel_score")),
  nav_panel("Denetim / Audit",
            layout_columns(value_box("max |rho| (231)", sprintf("%.3f", A$independence$max_abs_rho)),
                           value_box("Behaviour-only balanced acc.", sprintf("%.2f", A$models$behaviour_only$balanced_accuracy)),
                           value_box("All-columns accuracy", sprintf("%.1f%%", 100 * A$models$all_columns$accuracy))),
            uiOutput("finding"), plotOutput("heat", height = "520px")),
  nav_panel("Tahmin / Prediction", tableOutput("models"), plotOutput("weights", height = "320px")),
  nav_panel("Keşif / Explore", uiOutput("eff"), plotOutput("box", height = "420px"))
)

server <- function(input, output, session) {
  tr <- reactive(input$lang == "tr")
  output$finding <- renderUI(p(if (tr())
    "22 sayısal sütunun hiçbiri bir diğeriyle ilişkili değil; 13 puan sütununun 13'ü de 0–10 arasında düzgün. Veri, bağımsız rastgele çekilişlerden oluşan bir üreteç çıktısı."
    else "None of the 22 numeric columns is related to another; all 13 score columns are uniform on 0–10. The data are independent random draws."))
  output$heat <- renderPlot({
    C <- cor(d[num], method = "spearman"); par(mar = c(10, 10, 1, 1))
    image(seq_along(num), seq_along(num), C, zlim = c(-1, 1), col = hcl.colors(21, "Blue-Red 3", rev = TRUE), axes = FALSE, xlab = "", ylab = "")
    axis(1, seq_along(num), num, las = 2, cex.axis = .7); axis(2, seq_along(num), num, las = 2, cex.axis = .7)
  })
  output$models <- renderTable(data.frame(model = names(A$models), do.call(rbind, lapply(A$models, as.data.frame))), digits = 4)
  output$weights <- renderPlot({
    w <- unlist(A$label_rule$weights_online_vs_store); par(mar = c(4, 13, 1, 1))
    barplot(rev(w), horiz = TRUE, las = 1, col = "#2d6a8f", border = NA, xlab = "weight (online − store)")
  })
  output$eff <- renderUI({
    H <- kruskal.test(d[[input$col]], d$shopping_preference)$statistic / (nrow(d) - 1)
    h4(sprintf("Kruskal ε² = %.3f  (%s)", H, if (tr()) "0,01 küçük · 0,06 orta · 0,14 büyük" else "0.01 small · 0.06 medium · 0.14 large"))
  })
  output$box <- renderPlot(boxplot(d[[input$col]] ~ d$shopping_preference, col = c("#9fb7c9", "#2d6a8f", "#c9b38f"), xlab = "", ylab = input$col))
}

shinyApp(ui, server)
